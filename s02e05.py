import os
import requests
import sys
from bs4 import BeautifulSoup
import json
from openai import OpenAI
import base64
from pathlib import Path
from urllib.parse import urlparse, urljoin
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from modules.ImageRecognizer import ImageRecognizer
from modules.Transcriber import Transcriber




class Utils:

           
    @staticmethod
    def fetch_webpage(url):
        """Pobiera stronę internetową i zwraca obiekt BeautifulSoup."""
        try:
            response = requests.get(url)
            response.raise_for_status()  # Sprawdza, czy nie wystąpił błąd HTTP
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup
        except requests.RequestException as e:
            print(f"Error fetching webpage: {e}")
            return None

    @staticmethod
    def save_file(url, folder):
        """Pobiera plik z URL-a i zapisuje go lokalnie."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            file_name = os.path.basename(url)
            file_path = os.path.join(folder, file_name)
            os.makedirs(folder, exist_ok=True)
            with open(file_path, 'wb') as file:
                file.write(response.content)
            print(f"File saved to {file_path}")
        except requests.RequestException as e:
            print(f"Error downloading file: {e}")

    @staticmethod
    def get_cached_or_generate_description(client, image_path, cache_dir):
        """Zwraca opis obrazu z cache lub generuje nowy opis."""
        cache_file = os.path.join(cache_dir, f"{os.path.basename(image_path)}.txt")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as file:
                return file.read()
        
        image_recognizer = ImageRecognizer(client)
        description = image_recognizer.recognize_image( image_path, 1000, "Opisz co widzisz na obrazie", "gpt-4o")
        
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, 'w') as file:
            file.write(description)
        
        return description

    @staticmethod
    def get_cached_or_transcribe_audio(client, audio_path, cache_dir):
        """Zwraca transkrypcję audio z cache lub generuje nową transkrypcję."""
        cache_file = os.path.join(cache_dir, f"{os.path.basename(audio_path)}.txt")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as file:
                return file.read()
        
        transcriber = Transcriber(client)
        transcription = transcriber.transcribe(audio_path)
        
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, 'w') as file:
            file.write(transcription)
        
        return transcription

class IndexHtml:
    
    def __init__(self, client):
        self.client = client

    def index_webpage(self, url, dir):
        # Pobierz stronę
        soup = Utils.fetch_webpage(url)
        if not soup:
            return "Failed to fetch webpage."

        # Tworzenie folderów
        image_folder = os.path.join(dir, r'temp\images')
        audio_folder = os.path.join(dir, r'temp\audio')
        os.makedirs(image_folder, exist_ok=True)
        os.makedirs(audio_folder, exist_ok=True)

        # Inicjalizacja text_content
        text_content = ""

        # Rekurencyjne przetwarzanie HTML
        for element in soup.descendants:
            if element.name is None:  # Tekst
                text_content += element.strip() + "\n"
            elif element.name == 'img' and element.get('src'):  # Obraz
                image_url = urljoin(url, element['src'])
                Utils.save_file(image_url, image_folder)
                image_path = os.path.join(image_folder, os.path.basename(image_url))
                description = Utils.get_cached_or_generate_description(self.client, image_path, image_folder)
                text_content += description + "\n"
            elif element.name == 'source' and element.get('type') == 'audio/mpeg' and element.get('src'):  # Audio
                audio_url = urljoin(url, element['src'])
                Utils.save_file(audio_url, audio_folder)
                audio_path = os.path.join(audio_folder, os.path.basename(audio_url))
                transcription = Utils.get_cached_or_transcribe_audio(self.client, audio_path, audio_folder)
                text_content += transcription + "\n"

        # Zwróć pełny tekst strony
        return text_content



class KnowledgeDb:
    
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key

    def prepare_knowledge_base(self, text_content):
        # Dzielenie tekstu na fragmenty
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text_chunks = text_splitter.split_text(text_content)

        # Tworzenie osadzeń (embeddings)
        embeddings_model = OpenAIEmbeddings(api_key=self.openai_api_key)
        embeddings = [embeddings_model.embed_text(chunk) for chunk in text_chunks]

        # Tworzenie przestrzeni FAISS
        faiss_index = FAISS()
        faiss_index.add_vectors(embeddings)

        # Tworzenie łańcucha QA
        chat_model = ChatOpenAI(api_key=self.openai_api_key)
        retrieval_qa = RetrievalQA(retriever=faiss_index, chat_model=chat_model)

        return retrieval_qa

class Answerer:
    
    def __init__(self, qa_chain, url_question, wrong_answers_file='wrong_answers.json'):
        self.qa_chain = qa_chain
        self.url_question = url_question
        self.wrong_answers_file = wrong_answers_file
        self.questions = self.load_questions()
        self.wrong_answers = self.load_wrong_answers()

    def load_questions(self):
        """Pobiera pytania z URL-a i zapisuje je w słowniku."""
        try:
            response = requests.get(self.url_question)
            response.raise_for_status()
            questions_data = response.text.splitlines()
            questions = {}
            for line in questions_data:
                if '=' in line:
                    q_id, question = line.split('=', 1)
                    questions[q_id.strip()] = question.strip()
            return questions
        except requests.RequestException as e:
            print(f"Error fetching questions: {e}")
            return {}

    def load_wrong_answers(self):
        """Wczytuje wcześniejsze błędne odpowiedzi z pliku."""
        if os.path.exists(self.wrong_answers_file):
            with open(self.wrong_answers_file, 'r') as file:
                return json.load(file)
        return {}

    def generate_answers(self):
        """Generuje odpowiedzi na pytania."""
        answers = {}
        for q_id, question in self.questions.items():
            prompt = question
            if q_id in self.wrong_answers:
                wrong_context = self.wrong_answers[q_id]
                prompt += f"\n\nPrevious wrong answers: {wrong_context}"
            
            # Generate answer using qa_chain
            answer = self.qa_chain.ask(prompt)
            answers[q_id] = answer
        return answers

def main():
    # Ładowanie zmiennych środowiskowych z pliku .env
    load_dotenv()

    # Przykład dostępu do zmiennej środowiskowej
    openai_api_key = os.getenv('OPENAI_API_KEY')
    url_article = os.getenv('URL_ARTICLE')
    dir = os.getenv('DIR')
    print(dir)

    client = OpenAI(api_key=openai_api_key)
    
    index_html = IndexHtml(client)
    text_content = index_html.index_webpage(url_article, dir)
    # print(text_content)
  
if __name__ == "__main__":
    main()


