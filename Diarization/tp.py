import whisper
from pyannote.audio import Pipeline
from dotenv import load_dotenv
import os

model = whisper.load_model("base")

# Transcribe the audio
result = model.transcribe(
    r"C:\Users\Anuj Bohra\Desktop\PostStudio\audioFiles\NK-EP-04.mp4"
)

# Extract the transcript with timestamps
segments = result["segments"]

# Step 2: Perform Speaker Diarization with Authentication
# Load environment variables from .env file
load_dotenv()

# Get Hugging Face token from environment variables
hf_token = os.getenv("HF_TOKEN")

# Initialize the diarization pipeline with the access token
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization", use_auth_token=hf_token
)

# Apply diarization
diarization = pipeline(
    r"C:\Users\Anuj Bohra\Desktop\PostStudio\audioFiles\NK-EP-04.mp4"
)


# Step 3: Combine Transcription and Diarization
# Function to find speaker at a given timestamp
def find_speaker(timestamp, diarization):
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if turn.start <= timestamp <= turn.end:
            return speaker
    return "Unknown"


# Combine transcription with speaker labels
for segment in segments:
    speaker = find_speaker(segment["start"], diarization)
    print(
        f"{segment['start']:.2f} - {segment['end']:.2f}: Speaker {speaker}: {segment['text']}"
    )
