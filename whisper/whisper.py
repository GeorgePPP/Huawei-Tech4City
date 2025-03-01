# !pip install --q git+https://github.com/m-bain/whisperx.git
import whisperx

def get_diar_segment(audio_file):
    device = "cuda" 
    batch_size = 16
    compute_type = "float16"

    # ASR
    model = whisperx.load_model("large-v3", device, compute_type=compute_type)

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)


    #Diarization
    diarize_model = whisperx.DiarizationPipeline(use_auth_token='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', device=device)

    diarize_segments = diarize_model(audio)

    result = whisperx.assign_word_speakers(diarize_segments, result)

    return result['segments'], diarize_segments