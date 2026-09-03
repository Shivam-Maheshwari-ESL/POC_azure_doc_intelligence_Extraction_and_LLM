import json
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from poc_azure_llm_data_extraction import call_llm_for_extraction


CONFIG_FILE = "./azure_config.json"


def load_credentials(config_path=CONFIG_FILE):
    """
    Load Azure Document Intelligence endpoint and key from a separate JSON file.

    Expected file format (azure_config.json):
    {
        "endpoint": "https://<your-resource-name>.cognitiveservices.azure.com/",
        "key": "<your-api-key>"
    }
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file '{config_path}' not found. "
            f"Please create it with 'endpoint' and 'key' fields."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    endpoint = config.get("endpoint")
    key = config.get("key")
    api_version = config.get("api_version", "2024-11-30")  # Default to a specific API version if not provided

    if not endpoint or not key:
        raise ValueError("Config file must contain 'endpoint' and 'key' fields.")

    return endpoint, key, api_version


def analyze_pdf_using_doc_intel(pdf_path, model_id="prebuilt-layout"):
    """
    Send the given PDF to Azure Document Intelligence and return the extracted result.
    """

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file '{pdf_path}' not found.")

    endpoint, key, api_version = load_credentials()

    client = DocumentIntelligenceClient(
        endpoint=endpoint, 
        credential=AzureKeyCredential(key),
        api_version=api_version
    )

    with open(pdf_path, "rb") as f:
        poller = client.begin_analyze_document(
            model_id=model_id,
            body=f,
            content_type="application/pdf",
        )

    result = poller.result()
    return result


def result_to_dict(result):
    """
    Convert the Azure Document Intelligence result object into a JSON-serializable dict.
    """
    return result.as_dict()


def save_result_as_json(result, output_path="extracted_data_doc_intell.json"):
    """
    Save the extracted data as a JSON file.
    """
    if not type(result) == dict:
        data = result_to_dict(result)
    else:
        data = result
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Extracted data saved to '{output_path}'")


def main(base_input_path=None, ADI_output_path=None, base_output_path=None, no_LLM_call=True):
    
    file_names_list = os.listdir(base_input_path)

    os.makedirs(ADI_output_path, exist_ok=True)
    os.makedirs(base_output_path, exist_ok=True)

    for file_name in file_names_list :

        input_pdf_path = os.path.join(base_input_path, file_name)
        output_ADI_json_path = os.path.join(ADI_output_path, f"output_ADI_{os.path.splitext(file_name)[0]}.json")
        output_json_path = os.path.join(base_output_path, f"output_ADI_LLM_{os.path.splitext(file_name)[0]}.json")

        try:
            print(f"\nProcessing '{file_name}'...")

            print("Calling Azure Document Intelligence for extraction...")
            result_doc_intel = analyze_pdf_using_doc_intel(input_pdf_path)

            raw_result_dict = result_doc_intel.as_dict()
            save_result_as_json(raw_result_dict, output_ADI_json_path)

            if no_LLM_call:
                print(f"Skipping LLM call for '{file_name}' as per configuration.")
                continue

            raw_json_context = json.dumps(raw_result_dict, ensure_ascii=False)

            print("Calling Azure LLM for further extraction...")
            result = call_llm_for_extraction(raw_json_context)

            print(f"Saving final extracted data to '{output_json_path}'...")
            save_result_as_json(result, output_json_path)

            print(f"\nExtracted Data for '{file_name}'")
        
        except Exception as e:
            print(f"Error processing '{file_name}': {e}")

            
if __name__ == "__main__":

    folder_to_process = "NEW_10_Documents"  # Change this to the folder you want to process

    input_pdf_path = f"./input_pdf/{folder_to_process}"
    ADI_output_path = f"./output_ADI_json/{folder_to_process}"
    base_output_path = f"./output_final_json_extracted/{folder_to_process}"
        
    main(base_input_path=input_pdf_path, ADI_output_path=ADI_output_path, base_output_path=base_output_path, no_LLM_call=True)
