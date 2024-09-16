import whisper
from pyannote.audio import Pipeline
from moviepy.editor import VideoFileClip
import os
from dotenv import load_dotenv
from pydub import AudioSegment

# Load environment variables from .env file
load_dotenv()

# Get Hugging Face token from environment variables
hf_token = os.getenv("HF_TOKEN")

# Paths
video_path = r"C:\Users\Anuj Bohra\Desktop\PostStudio\audioFiles\Untitled video - Made with Clipchamp.mp4"
audio_path = r"C:\Users\Anuj Bohra\Desktop\PostStudio\audioFiles\audio_extracted.wav"
chunk_duration_ms = 5 * 60 * 1000  # 5 minutes in milliseconds

# Extract audio from video using moviepy
try:
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(
        audio_path, codec="pcm_s16le"
    )  # Ensure audio is saved as WAV format
    print("Audio extracted successfully.")
except Exception as e:
    print(f"Error extracting audio: {e}")
    exit()

# Load the Whisper model
try:
    model = whisper.load_model("base")
    print("Whisper model loaded successfully.")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    exit()

# Split audio into chunks and process each chunk
audio_segment = AudioSegment.from_wav(audio_path)
audio_length_ms = len(audio_segment)
chunks = [
    audio_segment[i : i + chunk_duration_ms]
    for i in range(0, audio_length_ms, chunk_duration_ms)
]

# Initialize the diarization pipeline with the access token
try:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization", use_auth_token=hf_token
    )
    print("Pyannote pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading Pyannote pipeline: {e}")
    exit()

full_transcript = []
diarization_results = []

for i, chunk in enumerate(chunks):
    chunk_path = (
        f"C:\\Users\\Anuj Bohra\\Desktop\\PostStudio\\audioFiles\\chunk_{i}.wav"
    )
    chunk.export(chunk_path, format="wav")

    # Transcribe the audio chunk
    try:
        result = model.transcribe(chunk_path)
        full_transcript.extend(result["segments"])
        print(f"Transcription of chunk {i} completed.")
    except Exception as e:
        print(f"Error transcribing chunk {i}: {e}")
        continue

    # Apply diarization to the chunk
    try:
        diarization = pipeline(chunk_path)
        diarization_results.append(diarization)
        print(f"Speaker diarization of chunk {i} completed.")
    except Exception as e:
        print(f"Error performing speaker diarization on chunk {i}: {e}")
        continue


# Function to find speaker at a given timestamp
def find_speaker(timestamp, diarization):
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if turn.start <= timestamp <= turn.end:
            return speaker
    return "Unknown"


# Combine transcription with speaker labels
for segment in full_transcript:
    for diarization in diarization_results:
        speaker = find_speaker(segment["start"], diarization)
        if speaker != "Unknown":
            print(
                f"{segment['start']:.2f} - {segment['end']:.2f}: Speaker {speaker}: {segment['text']}"
            )
            break

print("Script completed successfully.")
