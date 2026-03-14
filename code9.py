import os
import sys
import json
import time
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix
from groq import Groq

# --- CONFIGURATION ---
GROQ_API_KEY = "gsk_S8qsebuY0rBULEToy1g" 

# These MUST match the headers in your Excel/Gold Standard file exactly
DISEASES = [
    'gastritis', 'ascites', 'colitis', 'liver_mass', 'pancreatitis', 
    'microhepatia', 'small_intestinal_obstruction', 'splenic_mass', 
    'splenomegaly', 'hepatomegaly'
]

def get_excel_file_path(prompt: str = "Enter path to Excel file: ") -> Path:
    while True:
        user_input = input(prompt).strip().strip('"').strip("'")
        if not user_input:
            continue
        path = Path(user_input).expanduser()
        if path.exists() and path.suffix.lower() in [".xlsx", ".xls", ".csv"]:
            return path
        print(f"Invalid file: {user_input}. Please ensure it exists and is an Excel or CSV file.")

def build_batch_labeling_prompt(reports, diseases):
    disease_list_str = ", ".join([f'"{d}"' for d in diseases])
    report_blocks = []
    for rpt in reports:
        report_blocks.append(
            f"--- CASE ID: {rpt.get('case_id')} ---\n"
            f"FINDINGS: {rpt.get('findings')}\n"
            f"CONCLUSION: {rpt.get('conclusion')}\n"
            f"RECOMMENDATION: {rpt.get('recommendation')}\n"
        )

    return f"""
### ROLE
### ROLE
You are a Board-Certified Veterinary Radiologist specializing in abdominal diagnostic imaging. Your goal is to classify reports with high diagnostic specificity.

### TASK
Perform a batch binary classification for the provided reports. For EACH report, determine if the target abdominal diseases are "Normal" or "Abnormal".

### ABDOMINAL CLASSIFICATION CRITERIA
Classify as **Abnormal** ONLY if the report explicitly describes findings consistent with these markers:
### MEDICAL KNOWLEDGE REFERENCE:
Use these specific markers to determine "Abnormal" status:
- **gastritis**: Wall thickening, mucosal irregularities, or abnormal contents suggesting inflammation.
- **ascites**: Peritoneal fluid, "loss of serosal detail," or a "ground glass" appearance.
- **colitis**: Thickening of colonic wall or abnormal gas patterns.
- **liver_mass**: Focal lesions, nodules, or masses within the liver.
- **pancreatitis**: Mottling in right cranial abdomen, widened gastroduodenal angle, or "corrugated" duodenum.
- **microhepatia**: Small liver; look for "cranial displacement of the gastric axis."
- **small_intestinal_obstruction**: Dilated bowel loops, foreign bodies, or "two-population" size distribution.
- **splenic_mass / splenomegaly**: Focal nodules (mass) or generalized enlargement beyond boundaries (splenomegaly).
- **hepatomegaly**: Generalized enlargement with rounded margins extending beyond the costal arch.

### TARGET DISEASES
[{disease_list_str}]

### RADIOLOGY REPORTS
{report_blocks}
(Note to LLM: Each block will contain CaseID, Findings, Conclusions, and Recommendations)

### OUTPUT REQUIREMENTS
Return ONLY a valid JSON array of objects. Do not include markdown formatting or conversational text outside the JSON.
IMPORTANT NOTES
----------------------------------------

• Absence of a disease name does NOT mean Normal.
• Many diseases must be inferred from radiologic signs.
• Be conservative but detect subtle abnormalities.


UNCERTAINTY HANDLING SHOULD BE DONE VERY CARFULL USING THE MODEL KNOWLEDGE OF RADIOLOGY REPORTS LIKE A DOCTOR.

## STRICT EVIDENCE RULE

Primary Directive: Do NOT classify as 'Abnormal' based on minor, incidental, or non-specific findings that the radiologist notes as clinically insignificant.

Threshold for Abnormal: Only classify as 'Abnormal' if there is a definitive diagnostic statement or clear radiological markers.

Ambiguity Handling: If the report uses terms like 'possible,' 'differential includes,' or 'cannot rule out' WITHOUT supporting visual evidence in the Findings section, default to Normal."



TARGET DISEASES
[{disease_list_str}]

RADIOLOGY REPORTS
{"".join(report_blocks)}

OUTPUT REQUIREMENTS

Return ONLY a valid JSON array.

Each object must contain:
• "report_id"
• one key for each disease

Each disease must have ONLY one value:
"Normal" or "Abnormal"

OUTPUT FORMAT EXAMPLE:
{{
  "results": [
    {{
      "CaseID": "12345",
      "gastritis": "Normal",
      "ascites": "Abnormal",
      ...
    }}
  ]
}}
""".strip()

def call_groq_batch(client, model_name, prompt, max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_name,
                temperature=0.0,
                response_format={"type": "json_object"} 
            )
            return json.loads(chat_completion.choices[0].message.content)
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return None

def label_reports_from_excel(excel_path: Path, client: Groq, model_name: str, diseases, batch_size: int = 3):
    # Load file (supports both CSV and Excel)
    if excel_path.suffix.lower() == ".csv":
        df = pd.read_csv(excel_path)
    else:
        df = pd.read_excel(excel_path)
    
    col_findings = "Findings (original radiologist report)"
    col_concl = "Conclusions (original radiologist report)"
    col_recom = "Recommendations (original radiologist report)"
    col_id = "CaseID"

    # Initialize columns
    for d in diseases:
        df[d] = "Normal" 

    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        batch_df = df.iloc[start:end]
        
        reports_for_ai = []
        for _, row in batch_df.iterrows():
            reports_for_ai.append({
                "case_id": str(row.get(col_id, "")),
                "findings": str(row.get(col_findings, "")),
                "conclusion": str(row.get(col_concl, "")),
                "recommendation": str(row.get(col_recom, ""))
            })

        prompt = build_batch_labeling_prompt(reports_for_ai, diseases)
        response = call_groq_batch(client, model_name, prompt)

        if response and "results" in response:
            results = response["results"]
            for i, res in enumerate(results):
                idx = start + i
                if idx < len(df):
                    for d in diseases:
                        df.at[idx, d] = res.get(d, "Normal")
        
        print(f"Processed {end}/{len(df)} reports...")
    return df

def generate_confusion_matrix(llm_df, gold_df, diseases):
    results = []
    for disease in diseases:
        if disease not in llm_df.columns or disease not in gold_df.columns:
            print(f"Skipping {disease}: Column missing in one of the files.")
            continue

        y_pred = llm_df[disease].astype(str).str.lower().map({'abnormal': 1, 'normal': 0}).fillna(0)
        y_true = gold_df[disease].astype(str).str.lower().map({'abnormal': 1, 'normal': 0}).fillna(0)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        total = tp + tn + fp + fn
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision   = tp / (tp + fp) if (tp + fp) > 0 else 0
        accuracy    = (tp + tn) / total if total > 0 else 0
        
        results.append({
            "Condition": disease,
            "TP": tp, "FN": fn, "TN": tn, "FP": fp,
            "Sensitivity (%)": round(sensitivity * 100, 2),
            "Precision (%)": round(precision * 100, 2),
            "Accuracy (%)": round(accuracy * 100, 2)
        })
    return pd.DataFrame(results)

def main():
    api_key = GROQ_API_KEY
    groq_client = Groq(api_key=api_key)
    groq_model = "llama-3.3-70b-versatile" 

    input_path = get_excel_file_path("Path to INPUT File: ")
    labeled_df = label_reports_from_excel(input_path, groq_client, groq_model, DISEASES)
    
    output_path = input_path.parent / "abdominal_labels_output.xlsx"
    labeled_df.to_excel(output_path, index=False)
    print(f"Labeling complete. Saved to {output_path}")

    gold_path = get_excel_file_path("Path to GOLD STANDARD File: ")
    gold_df = pd.read_excel(gold_path) if gold_path.suffix != ".csv" else pd.read_csv(gold_path)
    
    metrics_df = generate_confusion_matrix(labeled_df, gold_df, DISEASES)
    metrics_path = input_path.parent / "abdominal_metrics_summary.xlsx"
    metrics_df.to_excel(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
