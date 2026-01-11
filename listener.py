import torch
from faster_whisper import WhisperModel
import time

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Listener:
    """
    Recognize speech using Faster_Whisper
    """
    def __init__(self):
        """ Initialize speech recognition unit """
        self.model_name = "large-v3" # large-v3-turbo
        self.speech_model = WhisperModel(self.model_name, device=DEVICE)
        print(f"Speech recognition model loaded to device: {DEVICE}")

    def recognize(self, afile: str) -> str:
        """ Recognize text from an audio file """
        print("Device:", self.speech_model.model.device, " file:", afile)
        start = time.time()
        segments, info = self.speech_model.transcribe(afile)
        end = time.time()
        text = " ".join([segment.text for segment in segments])
        end2 = time.time()
        print(f"Length: {type(text)}, recognition time: {end - start:.2f}s, text joining: {end2-end:.2f}s")
        print(f"Transcription: {text}")
        return text