import os
import sys
import json
import time
import re
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix
from groq import Groq

# --- CONFIGURATION ---
# Paste your Groq API key between the quotes below
GROQ_API_KEY = "gsk_uO5uUMUIr7kenz7rSgzNWGdyb3FYzCw5a4AqIZYjs6ysAiNQH" 

DISEASES = [
    "perihilar_infiltrate", "pneumonia", "bronchitis", "interstitial",
    "diseased_lungs", "hypo_plastic_trachea", "cardiomegaly",
    "pulmonary_nodules", "pleural_effusion", "rtm",
    "focal_caudodorsal_lung", "focal_perihilar", "pulmonary_hypoinflation",
    "right_sided_cardiomegaly", "pericardial_effusion", "bronchiectasis",
    "pulmonary_vessel_enlargement", "left_sided_cardiomegaly",
    "thoracic_lymphadenopathy", "esophagitis"
]

def get_excel_file_path(prompt: str = "Enter path to Excel file: ") -> Path:
    while True:
        user_input = input(prompt).strip().strip('"').strip("'")
        if not user_input:
            continue
        path = Path(user_input).expanduser()
        if path.exists() and path.suffix.lower() in [".xlsx", ".xls"]:
            return path
        print(f"Invalid file: {user_input}. Please ensure it exists and is an Excel file.")

def build_batch_labeling_prompt(reports, diseases):
    disease_list_str = ", ".join([f'"{d}"' for d in diseases])
    report_blocks = []
    for i, rpt in enumerate(reports, start=1):
        report_blocks.append(
            f"--- REPORT {i} ---\n"
            f"FINDINGS: {rpt.get('findings')}\n"
            f"CONCLUSION: {rpt.get('conclusion')}\n"
            f"RECOMMENDATION: {rpt.get('recommendation')}\n"
        )

    return f"""
Act as a board-certified veterinary radiologist. Perform a batch binary classification.
OBJECTIVE: Classify each disease as "Normal" or "Abnormal".

TARGET DISEASES:
[{disease_list_str}]

REPORTS:
{"".join(report_blocks)}

LOGIC:
- Abnormal: Mentions of pathology, "cannot rule out", or "suggestive of".
- Normal: Stated unremarkable, no evidence, or not mentioned at all.

OUTPUT:
Return ONLY a raw JSON array of objects. Each object must have "report_id" and the disease keys.
No markdown, no talk.
""".strip()

def call_groq_batch(client, model_name, prompt, max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_name,
                temperature=0.0,
                # Groq specific: ensure JSON response
                response_format={"type": "json_object"} 
            )
            
            text_output = chat_completion.choices[0].message.content
            return json.loads(text_output)

        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return None

def label_reports_from_excel(excel_path: Path, client: Groq, model_name: str, diseases, batch_size: int = 5):
    df = pd.read_excel(excel_path)
    
    # Precise Column Mapping
    col_findings = "Findings (original radiologist report)"
    col_concl = "Conclusions (original radiologist report)"
    col_recom = "Recommendations (original radiologist report)"

    for d in diseases:
        df[d] = None

    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        batch_df = df.iloc[start:end]
        
        reports_for_ai = []
        for _, row in batch_df.iterrows():
            reports_for_ai.append({
                "findings": str(row.get(col_findings, "")),
                "conclusion": str(row.get(col_concl, "")),
                "recommendation": str(row.get(col_recom, ""))
            })

        prompt = build_batch_labeling_prompt(reports_for_ai, diseases)
        results_data = call_groq_batch(client, model_name, prompt)

        # Handle various JSON structures LLMs might return
        if results_data:
            # If the LLM wraps the array in a key (like "reports"), extract it
            results = results_data if isinstance(results_data, list) else list(results_data.values())[0]
            
            for i, res in enumerate(results):
                idx = start + i
                if idx < len(df):
                    for d in diseases:
                        val = res.get(d, "Normal")
                        df.at[idx, d] = val
        
        print(f"Processed {end}/{len(df)}...")
        time.sleep(1) 
    return df

def generate_confusion_matrix(llm_df, gold_df, diseases):
    results = []

    for disease in diseases:

        y_pred = llm_df[disease].str.lower().map({'abnormal': 1, 'normal': 0}).fillna(0)
        y_true = gold_df[disease].str.lower().map({'abnormal': 1, 'normal': 0}).fillna(0)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        total = tp + tn + fp + fn
        if total == 0:
            continue

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision   = tp / (tp + fp) if (tp + fp) > 0 else 0
        accuracy    = (tp + tn) / total
        f1_score    = (2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0
        prevalence  = (tp + fn) / total
        balanced_acc = (sensitivity + specificity) / 2

        results.append({
            "Condition": disease,
            "TP": tp,
            "FN": fn,
            "TN": tn,
            "FP": fp,
            "Prevalence (%)": round(prevalence * 100, 2),
            "Sensitivity (%)": round(sensitivity * 100, 2),
            "Specificity (%)": round(specificity * 100, 2),
            "Precision (%)": round(precision * 100, 2),
            "Accuracy (%)": round(accuracy * 100, 2),
            "F1 Score (%)": round(f1_score * 100, 2),
            "Balanced Accuracy (%)": round(balanced_acc * 100, 2)
        })

    return pd.DataFrame(results)

def main():
    # Priority 1: Use the hardcoded key. Priority 2: Use environment variable.
    api_key = GROQ_API_KEY if GROQ_API_KEY != "gsk_your_actual_key_here" else os.environ.get("GROQ_API_KEY")

    if not api_key:
        print("Error: No Groq API Key found. Please paste it into the script or set GROQ_API_KEY env var.")
        sys.exit(1)

    groq_client = Groq(api_key=api_key)
    # Most powerful versatile model on Groq currently
    groq_model = "openai/gpt-oss-120b" 

    # 1. Labeling
    input_path = get_excel_file_path("Path to INPUT Excel: ")
    labeled_df = label_reports_from_excel(input_path, groq_client, groq_model, DISEASES)
    
    output_path = input_path.parent / "llm_labels_output.xlsx"
    labeled_df.to_excel(output_path, index=False)
    print(f"Labeling complete. Saved to {output_path}")

    # 2. Evaluation
    gold_path = get_excel_file_path("Path to GOLD STANDARD Excel: ")
    gold_df = pd.read_excel(gold_path)
    
    metrics_df = generate_confusion_matrix(labeled_df, gold_df, DISEASES)
    metrics_path = input_path.parent / "metrics_summary.xlsx"
    metrics_df.to_excel(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
