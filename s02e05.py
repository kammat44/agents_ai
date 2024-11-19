import os
import requests
import sys
sys.path.append("C:\\Users\\kamyk\\Documents\\01_PROJECTS\\AI_DEVS_3")
from bs4 import BeautifulSoup
import json
from openai import OpenAI
from urllib.parse import  urljoin
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from dotenv import load_dotenv
from modules.ImageRecognizer import ImageRecognizer
from modules.Transcriber import Transcriber
from modules.Responder import ReportSenderAnswerJson



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
        # Save text_content to context.txt in the specified directory        
        temp_dir = os.path.join(dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        context_file_path = os.path.join(temp_dir, 'context.txt')
        
        with open(context_file_path, 'w', encoding='utf-8') as file:
            file.write(text_content)
            
        return text_content



class KnowledgeDb:
    
    def __init__(self, api_key):
        self.api_key = api_key

    def prepare_knowledge_base(self, text_content):
        # Dzielenie tekstu na fragmenty
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        text_chunks = text_splitter.split_text(text_content)

        # Tworzenie obiektów dokumentów z fragmentów tekstu
        documents = [Document(page_content=chunk) for chunk in text_chunks]

        # Tworzenie przestrzeni FAISS z dokumentów
        vectorstore = FAISS.from_documents(documents=documents, embedding=OpenAIEmbeddings(api_key=self.api_key))

        # Tworzenie łańcucha QA
        llm = ChatOpenAI(api_key=self.api_key) 
        
        # https://smith.langchain.com/hub/rlm/rag-prompt
        # prompt = hub.pull("rlm/rag-prompt")
        
        prompt = ChatPromptTemplate.from_messages([("human", "Jesteś asystentem odpowiadającym na pytania. Użyj kontekstu, aby odpowiedzieć na pytanie. Użyj maksymalnie jednego zdania i utrzymaj odpowiedź zwięzłą i precyzyjną.  Question: {question} Context: {context} Answer:"),])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)


        qa_chain = (
            {
                "context": vectorstore.as_retriever() | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        return qa_chain

class Answerer:
    
    def __init__(self, qa_chain, url_question):
        self.qa_chain = qa_chain
        self.url_question = url_question
        self.questions = self.load_questions()

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

    def generate_answers(self):
        """Generuje odpowiedzi na pytania."""
        answers = {}
        print(self.questions)
        for q_id, question in self.questions.items():
            prompt = question        
            # Generate answer using qa_chain
            answer = self.qa_chain.invoke(prompt)
            answers[q_id.strip()] = answer.strip()
        return answers
    
    def generate_answers_without_qa(self):
        """Generuje odpowiedzi na pytania."""
        answers = {}
        print(self.questions)
        for q_id, question in self.questions.items():
            prompt = question        
            # Generate answer using qa_chain
            answer = self.qa_chain.invoke(prompt)
            answers[q_id.strip()] = answer.strip()
        return answers
    
class SimpleAnswerer:
    
    def __init__(self, url_question, client, context):
        self.client = client
        self.context = context
        self.url_question = url_question
        self.questions = self.load_questions()

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
    
    def generate_answers_without_qa(self):
        """Generuje odpowiedzi na pytania."""
        answers = {}
        print(self.questions)
        for q_id, question in self.questions.items():
            # Use GPT-4o to generate the answer
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. Use one sentence and keep the answer concise."},
                    {"role": "user", "content": self.context},
                    {"role": "user", "content": question},
                ],
                max_tokens=1000,
            )
            answer = response.choices[0].message.content.strip()
            answers[q_id.strip()] = answer
        return answers  

def main():
    # Ładowanie zmiennych środowiskowych z pliku .env
    load_dotenv()

    # Przykład dostępu do zmiennej środowiskowej
    openai_api_key = os.getenv('OPENAI_API_KEY')
    report_api_key = os.getenv('REPORT_API_KEY')

    url_article = os.getenv('URL_ARTICLE')
    url_question = os.getenv('URL_QUESTION') 
    url_answer = os.getenv('URL_ANSWER')   
    dir = os.getenv('DIR')


    client = OpenAI(api_key=openai_api_key)
    
    # index_html = IndexHtml(client)
    # text_content = index_html.index_webpage(url_article, dir)
    # # print(text_content)
    
    # knowledge_db = KnowledgeDb(api_key=openai_api_key)
    # qa_chain = knowledge_db.prepare_knowledge_base(text_content)
    
    # answerer = Answerer(qa_chain, url_question)
    # answers = answerer.generate_answers()
    # print(answers)
    
    
    #wersja z zastosowaniem simple answerera działa od strzała
    
    with open("C:\\Users\\kamyk\\Documents\\01_PROJECTS\\AI_DEVS_3\\aidevs3_s02e05\\temp\\context.txt", 'r', encoding='utf-8') as file:
        context = file.read()
    print(context)
    simple_answerer = SimpleAnswerer(url_question, client, context )
    answers = simple_answerer.generate_answers_without_qa()
    print(answers)
    report_sender = ReportSenderAnswerJson(report_api_key,"arxiv",url_answer)
    report_sender.send_report(answers)
    
if __name__ == "__main__":
    main()


