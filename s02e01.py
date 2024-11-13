import os
from pydub import AudioSegment
import whisper
from openai import AsyncOpenAI
import asyncio
import requests

class AudioTranscriber:
    def __init__(self, directory):
        self.directory = directory
        self.model = whisper.load_model("base")  # Load the Whisper model

    def convert_and_transcribe(self):
        for filename in os.listdir(self.directory):
            if filename.endswith(".m4a"):
                print(f"Processing file: {filename}")
                self.process_file(filename)

    def process_file(self, filename):
        file_path = os.path.join(self.directory, filename)
        wav_path = self.convert_to_wav(file_path)
        self.transcribe_audio(wav_path, filename)
        os.remove(wav_path)

    def convert_to_wav(self, file_path):
        audio = AudioSegment.from_file(file_path, format="m4a")
        wav_path = file_path.replace(".m4a", ".wav")
        audio.export(wav_path, format="wav")
        print(f"Converted {file_path} to {wav_path}")
        return wav_path

    def transcribe_audio(self, wav_path, original_filename):
        # Use Whisper to transcribe the audio
        result = self.model.transcribe(wav_path)
        text = result['text']
        self.save_transcription(text, original_filename)
        print(f"Transcription for {original_filename} completed.")

    def save_transcription(self, text, original_filename):
        text_file_path = os.path.join(self.directory, original_filename.replace(".m4a", ".txt"))
        with open(text_file_path, "w", encoding="utf-8") as text_file:
            text_file.write(text)
        print(f"Saved transcription to {text_file_path}")

class Answerer:
    def __init__(self, directory, api_key):
        self.directory = directory
        self.client = AsyncOpenAI(api_key=api_key)

    def read_txt_files(self):
        texts = []
        for filename in os.listdir(self.directory):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.directory, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    texts.append(file.read())
        return texts

    async def build_common_context(self, texts):
        # Use the LLM to build a common context from the texts
        try:
            response = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Translate all files in english. Add content of the files as they are"},
                    {"role": "user", "content": " ".join(texts)},
                ],
                model='gpt-4o',
                max_tokens=2000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error using LLM to build context: {e}")
            return "Error in building context."

    async def answer_question(self, context, question):
        # Use the LLM to answer the question based on the context
        try:
            response = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": " Context doesn't contain direct answer, LLM reasoning is needed.Use the details and check internal LLM knowledge to find the answer. Analyze context and question several times to localize the city, then university, then institute, then street name.Describe all your reasoning. Thirst think at loud then write the answer."},
                    {"role": "user", "content": context},
                    {"role": "user", "content": question},
                ],
                model='gpt-4o',
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error using LLM to answer question: {e}")
            return "Error in answering question."

    async def extract_question(self, system_prompt, context, question):
        # Use the LLM to answer the question based on the context
        try:
            response = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context},
                    {"role": "user", "content": question},
                ],
                model='gpt-4o',
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error using LLM to answer question: {e}")
            return "Error in answering question."

class ReportSender:
    def __init__(self, api_key):
        self.api_key = api_key
        #ADD URL !!!
        self.report_url = "HERE URL TO THE REPORT API"

    def send_report(self, answer):
        # Ensure the answer is a single word
        answer = answer.split()[0]
        json_structure = {
            "task": "mp3",
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

if __name__ == "__main__":
    
    #ADD PATH TO THE MP3 FILES !!!
    
    directory = r"HERE PATH TO THE MP3 FILES"
    
    # transcriber = AudioTranscriber(directory)
    # transcriber.convert_and_transcribe()

    # Initialize the Answerer
    api_key = os.environ.get("OPENAI_API_KEY")
    answerer = Answerer(directory, api_key)
    report_api_key = os.environ.get("REPORT_API_KEY")
    # Read text files and build context
    texts = answerer.read_txt_files()
    context = asyncio.run(answerer.build_common_context(texts))
    # Answer the prompt question
    question = "Write the name of the street, where the institute is located, where the professor teaches."
    answer = asyncio.run(answerer.answer_question(context, question))
    single_word_answer = asyncio.run(answerer.extract_question("Extract single word answer from the context", answer, "What is the street name?"))
    print("Answer:", single_word_answer)

    # Send the answer as a report
    report_sender = ReportSender(report_api_key)
    report_sender.send_report(single_word_answer)