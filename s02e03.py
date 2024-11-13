import os
import requests
from openai import OpenAI

class PromptReader:
    def __init__(self, url):
        self.url = url

    def get_prompt(self):
        response = requests.get(self.url)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()
        return data.get('description', '')

class ImageGenerator:
    def __init__(self, client):
        self.client = client

    def generate_image(self, prompt):
        response = self.client.images.generate(
            model="dall-e-3",
            prompt=prompt,    
            size="1024x1024",
            quality="standard",
            n=1
        )
        return response.data[0].url

class ReportSender:
    def __init__(self, api_key):
        self.api_key = api_key
        
        #ADD URL !!!
        self.report_url = "HERE URL TO THE REPORT API"

    def send_report(self, answer):
        # Ensure the answer is a single word
        answer = answer.split()[0]
        json_structure = {
            "task": "robotid",
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

# Usage
api_key = os.environ.get("OPENAI_API_KEY")
report_api_key = os.environ.get("REPORT_API_KEY")
# Usage

#ADD URL !!!
prompt_reader = PromptReader(f"https://URL/data/{report_api_key}/robotid.json")

prompt = prompt_reader.get_prompt()
print(prompt)
client = OpenAI(api_key=api_key)
image_generator = ImageGenerator(client)
image_url = image_generator.generate_image(prompt)
print(image_url)

# Send the answer as a report
report_sender = ReportSender(report_api_key)
report_sender.send_report(image_url)
