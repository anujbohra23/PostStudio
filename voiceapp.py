import streamlit as st
import numpy as np
import torch
from transformers import pipeline
import librosa
import io
import srt
from datetime import timedelta
import json

# Load your pre-trained model from Hugging Face
model_name_or_path = "bohraanuj23/whisper-marathi-small"
asr_pipeline = pipeline("automatic-speech-recognition", model=model_name_or_path)

# Streamlit app title
st.title("Voice Recognition with Hugging Face and Streamlit")

# File uploader
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    # Read the file as bytes
    file_bytes = uploaded_file.read()

    # Use librosa or another library to process the audio file
    import librosa
    import io

    y, sr = librosa.load(io.BytesIO(file_bytes), sr=None)

    # Convert the audio to the required format
    audio_array = np.array(y, dtype=np.float32)

    # Perform speech recognition using the pipeline
    result = asr_pipeline(audio_array)

    # Display the result
    st.write("Transcription:")
    st.write(result["text"])

    # Optionally save the results to various formats
    st.download_button("Download TXT", result["text"], file_name="transcription.txt")

    # Generate other formats if necessary
    import srt
    from datetime import timedelta

    # Example to generate SRT format (you'll need to implement proper segmentation)
    subtitles = [
        srt.Subtitle(
            index=1,
            start=timedelta(seconds=0),
            end=timedelta(seconds=len(y) / sr),
            content=result["text"],
        )
    ]
    srt_output = srt.compose(subtitles)
    st.download_button("Download SRT", srt_output, file_name="transcription.srt")

    # Generate VTT format
    vtt_output = srt_output.replace(" --> ", " --> ").replace(",", ".")
    st.download_button("Download VTT", vtt_output, file_name="transcription.vtt")

    # Example to generate TSV format
    tsv_output = f"start\tend\ttext\n0\t{len(y) / sr}\t{result['text']}"
    st.download_button("Download TSV", tsv_output, file_name="transcription.tsv")

    # Example to generate JSON format
    import json

    json_output = json.dumps({"transcription": result["text"]}, indent=4)
    st.download_button("Download JSON", json_output, file_name="transcription.json")
