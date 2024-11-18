class Transcriber:
    def __init__(self, client):
        self.client = client  

    def transcribe(self, file_path, language="pl"):
        audio_file= open(file_path, "rb")
        transcription = self.client.audio.transcriptions.create(
            language=language,
            model="whisper-1", 
            file=audio_file
        )
        return transcription.text
