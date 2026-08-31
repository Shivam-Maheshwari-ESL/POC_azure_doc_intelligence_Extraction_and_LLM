import json
import os
import re
from openai import AzureOpenAI

model = "gpt-5.6-sol" 
model = "gpt-5.6-terra"  
model = "gpt-5.6-luna"  
model = "gpt-5.4-mini"  
model = "gpt-5.4"  

API_KEY_FILE = "./AzureOpeapiKeys.txt"


# ------------------------------------
# SSI Prompt
# ------------------------------------
SSI_PROMPT = """
You are an expert financial document extraction engine specializing in Standard Settlement Instructions (SSI).

OBJECTIVE

Identify and extract every SSI present in the Azure Document Intelligence JSON representation of the PDF with the highest possible accuracy and deterministic results.

GENERAL EXTRACTION PRINCIPLES

1. Extract ONLY information explicitly present in the JSON.
2. Never infer, predict, estimate, normalize, complete, or generate missing values.
3. Never use industry knowledge to populate fields that are not explicitly stated.
4. If a value cannot be clearly read, return null and explain why in extraction_notes.
5. Do not merge information from different SSI records unless the document explicitly indicates they belong together.
6. Preserve original values exactly as they appear.
7. Preserve original capitalization, spacing, punctuation, account identifiers, SWIFT codes, and bank names.
8. Preserve line breaks within multi-line table cells whenever they contain meaningful information.
9. Extraction quality takes precedence over formatting preferences.
10. Produce the same result for the same JSON every time.

SSI IDENTIFICATION RULES

An SSI is a complete settlement instruction set.

Identify SSI boundaries using explicit evidence such as:
- Currency
- Settlement route
- Beneficiary details
- Receiving bank
- Correspondent bank
- Intermediary bank
- Custodian information
- Agent bank information
- SWIFT/BIC information
- Account information
- Settlement instruction sections
- Bank-specific SSI headings

A JSON may contain:
- One SSI
- Multiple SSIs
- Several currencies under the same client
- Multiple settlement routes
- Multiple beneficiaries

Create a separate SSI object whenever the document presents a distinct settlement instruction.

EXTRACTION RULES

For each SSI:

1. Capture every explicitly available field.
2. Do not restrict extraction to predefined attributes.
3. Dynamically create field names from document labels.
4. Preserve hierarchical relationships when present.
5. Preserve table structures when they contain SSI data.
6. Preserve multi-line content exactly.
7. Include all related notes, conditions, settlement remarks, and special instructions.

CONFIDENCE RULES

For every extracted field capture:

- value
- source_text
- source_page
- confidence

Confidence values:
- HIGH = clearly readable and explicitly stated
- MEDIUM = present but formatting/OCR ambiguity exists
- LOW = partially visible or uncertain

Never upgrade confidence based on assumptions.

OUTPUT FORMAT

{
  "document_metadata": {
    "total_pages": <number if known>,
    "processing_mode": "native_text|ocr|mixed"
  },
  "ssi_records": [
    {
      "ssi_id": "<generated sequential identifier>",
      "fields": {
        "<field_name>": {
          "value": "...",
          "source_text": "...",
          "source_page": 1,
          "confidence": "HIGH"
        }
      },
      "tables": [
        {
          "table_name": "...",
          "headers": [],
          "rows": []
        }
      ],
      "extraction_notes": []
    }
  ]
}

HANDLING UNKNOWN STRUCTURES

Because SSI formats vary across banks:

- Do not assume a universal template.
- Do not assume field names.
- Dynamically discover labels and attributes.
- Include all available information belonging to a specific SSI.
- If a bank-specific field appears, retain it exactly as found.
- If a field appears only inside a table, preserve the table relationship.

HALLUCINATION PREVENTION

Before finalizing output:

1. Verify every extracted value exists in the document.
2. Remove any value not directly supported by source text.
3. Remove derived, inferred, or calculated values.
4. Ensure every field can be traced to a specific page and source snippet.
5. If evidence cannot be shown, exclude the value.

Return valid JSON only.
"""


def load_llm_credentials(config_path=API_KEY_FILE):
    """
    Parse endpoint / api-key / api-version / deployment-name out of the
    '${VAR:default}' style config file (AzureOpeapiKeys.txt).
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found.")

    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    def extract(field):
        match = re.search(rf"{field}:\s*\$\{{[^:]+:([^}}]+)\}}", content)
        if not match:
            raise ValueError(f"Could not find '{field}' in '{config_path}'.")
        return match.group(1).strip()

    endpoint = extract("endpoint")
    api_key = extract("api-key")
    api_version = extract("api-version")
    deployment_name = extract("deployment-name")

    return endpoint, api_key, api_version, deployment_name

 
def call_llm_for_extraction(extracted_data_to_process=None):
    endpoint, api_key, api_version, deployment_name = load_llm_credentials()

    deployment_name = model

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )

    
    prompt = f"""
    Extract all SSI entries from the provided Azure Document Intelligence JSON output.

    Requirements:
    - Extract every SSI present in the document.
    - Do not infer or generate missing values.
    - Return results in valid JSON format.
    - Ensure maximum accuracy, consistency, and determinism.

    Input JSON:
    {extracted_data_to_process}
    """

    response = client.chat.completions.create(
        model=deployment_name,
        temperature=0,
        # reasoning_effort="high",
        messages=[
            {"role": "system", "content": SSI_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    return json.loads(content)


 