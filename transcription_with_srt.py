import os
import json
import whisper
import gradio as gr


model = whisper.load_model("small")
os.makedirs("transcriptions", exist_ok=True)


def transcribe(audio):
    # trimming the audio as per the model
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # to detect the lanf
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    transcription_text = result.text
    base_filename = "transcriptions/transcription"
    txt_file = f"{base_filename}.txt"
    json_file = f"{base_filename}.json"
    srt_file = f"{base_filename}.srt"
    vtt_file = f"{base_filename}.vtt"

    with open(txt_file, "w") as file:
        file.write(transcription_text)

    with open(json_file, "w") as file:
        json.dump({"text": transcription_text}, file, indent=4)

    def save_as_subtitle_format(filename, text, format_func):
        with open(filename, "w") as file:
            file.write(format_func(text))

    srt_format = lambda text: f"1\n00:00:00,000 --> 00:00:30,000\n{text}\n\n"
    vtt_format = lambda text: f"WEBVTT\n\n1\n00:00:00.000 --> 00:00:30.000\n{text}\n\n"

    save_as_subtitle_format(srt_file, transcription_text, srt_format)
    save_as_subtitle_format(vtt_file, transcription_text, vtt_format)

    return transcription_text, txt_file, json_file, srt_file, vtt_file


# gradio
gr.Interface(
    title="OpenAI Whisper ASR Gradio Web UI",
    fn=transcribe,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs=[
        gr.Textbox(label="Transcription"),
        gr.File(label="TXT File"),
        gr.File(label="JSON File"),
        gr.File(label="SRT File"),
        gr.File(label="VTT File"),
    ],
    live=True,
).launch(share=True)
