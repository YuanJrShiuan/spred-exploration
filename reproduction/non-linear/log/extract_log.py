import os

def extract_report_from_log(log_file_path, output_file_path):
    """
    Extract the content between "Begin of Report" and "End of Report" from a single log file, 
    and write it to the specified output file.

    Parameters:
        log_file_path (str): The path to the log file.
        output_file_path (str): The path to the output file.
    """
    try:
        with open(log_file_path, 'r', encoding='utf-8') as log_file:
            lines = log_file.readlines()
        
        in_report = False
        report_content = []
        
        for line in lines:
            if "---- Begin of Report ----" in line:
                in_report = True
            elif "---- End of Report ----" in line:
                in_report = False
                report_content.append(line)
            if in_report:
                report_content.append(line)
        
        if report_content:
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.writelines(report_content)
            print(f"Report content has been extracted to {output_file_path}.")
        else:
            print(f"No report content found in {log_file_path}.")
    except Exception as e:
        print(f"Error processing file {log_file_path}: {e}")


def process_all_logs(log_folder):
    """
    Traverse all .log files in the specified folder and extract report content.

    Parameters:
        log_folder (str): The path to the folder containing log files.
    """
    if not os.path.exists(log_folder):
        print(f"Folder {log_folder} does not exist.")
        return

    for filename in os.listdir(log_folder):
        if filename.endswith(".log") and "report" not in filename:
            log_file_path = os.path.join(log_folder, filename)
            model_name = os.path.splitext(filename)[0]
            output_file_path = os.path.join(log_folder, f"{model_name}_report.log")
            print(f"Processing file: {log_file_path}")
            extract_report_from_log(log_file_path, output_file_path)


if __name__ == "__main__":
    log_folder = r"non-linear/log"
    process_all_logs(log_folder)