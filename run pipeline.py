import json
import os

from poc_azure_doc_intel_data_extraction_Standalone import analyze_pdf_using_doc_intel, save_result_as_json
from poc_azure_llm_data_extraction import call_llm_for_extraction



def main():
    base_input_path = "./input_pdf"
    base_output_path = "./output_final_json_extracted"
    file_names_list = os.listdir(base_input_path)
    for file_name in file_names_list :

        input_pdf_path = os.path.join(base_input_path, file_name)
        output_json_path = os.path.join(base_output_path, f"{os.path.splitext(file_name)[0]}.json")
        try:
            print(f"\nProcessing '{file_name}'...")

            print("Calling Azure Document Intelligence for extraction...")
            result_doc_intel = analyze_pdf_using_doc_intel(input_pdf_path)

            raw_result_dict = result_doc_intel.as_dict()
            raw_json_context = json.dumps(raw_result_dict, ensure_ascii=False)

            print("Calling Azure LLM for further extraction...")
            result = call_llm_for_extraction(raw_json_context)

            print(f"Saving final extracted data to '{output_json_path}'...")
            save_result_as_json(result, output_json_path)

            print(f"\nExtracted Data for '{file_name}'")
        
        except Exception as e:
            print(f"Error processing '{file_name}': {e}")
            
if __name__ == "__main__":
    main()
    print("END")

