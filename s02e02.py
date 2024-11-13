import os
import asyncio
from PIL import Image
from openai import AsyncOpenAI # Assuming OpenAI provides the GPT-4o Vision API
import base64

class ImageRecognizer:
    def __init__(self, api_key, directory_path):
        self.api_key = api_key
        self.directory_path = directory_path
        self.client = AsyncOpenAI(api_key=api_key)

    async def recognize_images(self):
        # List all .png files in the directory
        png_files = [f for f in os.listdir(self.directory_path) if f.endswith('.png')]
        descriptions = []  # Collect descriptions here
        
        for file_name in png_files:
            file_path = os.path.join(self.directory_path, file_name)
            
            # Open the image file and encode it in base64
            with open(file_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                
                # Call the GPT-4o Vision API
                response = await self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "What’s in this image? Focus on: geography, intersections of the streets, road numbers, landmarks addresses."},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=2000,
                )
                
                # Process the response
                description = response.choices[0].message.content.strip() 
                descriptions.append("Results for " + str(file_name) + ": " + str(description))
        
        # Save descriptions to a text file
        descriptions_text = " ".join(descriptions)
        
        #ADD PATH TO THE DESCRIPTIONS FILE !!!  
        
        descriptions_file_path = r'HERE PATH TO THE DESCRIPTIONS FILE'
        with open(descriptions_file_path, "w") as text_file:
            text_file.write(descriptions_text)
        
        return descriptions_file_path  # Return the path to the descriptions file

class Answerer:
    def __init__(self, directory, api_key):
        self.directory = directory
        self.client = AsyncOpenAI(api_key=api_key)

    
    async def build_common_context(self, texts):
        # Use the LLM to build a common context from the images
        try:
            response = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": " Add text files content as they are."},
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
                    {"role": "system", "content": "You are an expert at Polish geography. Focus on streets names, road numers, and building locations. Describe all your reasoning at loud before writing the answer. One image is incorrect, please ignore it. Perform several iterations which city contains alle the streets needed. "},
                    {"role": "user", "content": context},
                    {"role": "user", "content": question},
                ],
                model='gpt-4o',
                max_tokens=2000
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
            
if __name__ == "__main__":            
    # Usage
    
    #ADD PATH TO THE IMAGES !!! 
    
    directory_path = r'HERE PATH TO THE IMAGES'
    
    api_key = os.environ.get("OPENAI_API_KEY")
    recognizer = ImageRecognizer(api_key, directory_path)
    
    # Run the image recognition and get the path to the descriptions file
    descriptions_file_path = asyncio.run(recognizer.recognize_images())
    
    # Read the image descriptions from the file
    with open(descriptions_file_path, "r") as text_file:
        image_descriptions = text_file.read()
    
    # Initialize Answerer and build context using the file content
    answerer = Answerer(directory_path, api_key)
    # context = asyncio.run(answerer.build_common_context(image_descriptions))
    # print(context)
    # Answer the prompt question
    question = "What city is on the images?"
    answer = asyncio.run(answerer.answer_question(image_descriptions, question))
    print(answer)
    # single_word_answer = asyncio.run(answerer.extract_question("Extract single word answer from the context", answer, "Write the name of city"))
    # print("Answer:", single_word_answer)
