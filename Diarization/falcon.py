import pvfalcon
import whisper

model = whisper.load_model("medium")
result = model.transcribe(
    r"C:\Users\Anuj Bohra\Desktop\PostStudio\audioFiles\MarathiAudioSample.mp3"
)
transcript_segments = result["segments"]

falcon = pvfalcon.create(
    access_key="Fgzjtt2th8LXgFN+nf3y3hxoBG/L3ThMiKJZq7P6seS9kbr+aVWx8A=="
)
speaker_segments = falcon.process_file(
    r"C:\Users\Anuj Bohra\Desktop\PostStudio\audioFiles\MarathiAudioSample.mp3"
)


def segment_score(transcript_segment, speaker_segment):
    transcript_segment_start = transcript_segment["start"]
    transcript_segment_end = transcript_segment["end"]
    speaker_segment_start = speaker_segment.start_sec
    speaker_segment_end = speaker_segment.end_sec

    overlap = min(transcript_segment_end, speaker_segment_end) - max(
        transcript_segment_start, speaker_segment_start
    )
    overlap_ratio = overlap / (transcript_segment_end - transcript_segment_start)
    return overlap_ratio


for t_segment in transcript_segments:
    max_score = 0
    best_s_segment = None
    for s_segment in speaker_segments:
        score = segment_score(t_segment, s_segment)
        if score > max_score:
            max_score = score
            best_s_segment = s_segment

    if best_s_segment is not None:
        print(f"Speaker {best_s_segment.speaker_tag}: {t_segment['text']}")
    else:
        print(f"No speaker segment found for transcript: {t_segment['text']}")
