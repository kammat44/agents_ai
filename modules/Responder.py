import json
import requests

class ReportSenderAnswerJson:
    def __init__(self, api_key, task, report_url):
        self.api_key = api_key
        self.task = task
        self.report_url = report_url

    def send_report(self, answer):
        # Ensure the answer is a JSON object
        if isinstance(answer, str):
            try:
                answer = json.loads(answer)
            except json.JSONDecodeError:
                print("Error: The answer is not a valid JSON string.")
                return

        json_structure = {
            "task": self.task,
            "apikey": self.api_key,
            "answer": answer  # Send the JSON object directly
        }
        try:
            response = requests.post(self.report_url, json=json_structure)
            response.raise_for_status()
            print("Report sent successfully.")
            print("Response:", response.text)
        except Exception as e:
            print(f"Error sending report: {e}")
            print("Response:", response.text)

class ReportSenderAnswerString:
    def __init__(self, api_key, task, report_url):
        self.api_key = api_key
        self.task = task
        self.report_url = report_url

    def send_report(self, answer):
        answer = answer.split()[0]   
        json_structure = {
            "task": self.task,
            "apikey": self.api_key,
            "answer": answer  
        }
        try:
            response = requests.post(self.report_url, json=json_structure)
            response.raise_for_status()
            print("Report sent successfully.")
            print("Response:", response.text)
        except Exception as e:
            print(f"Error sending report: {e}")
            print("Response:", response.text)
            
         
