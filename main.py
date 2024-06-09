import os
import json

from whisper.whisper import get_diar_segment
from bert_personality.detect import personality_detection
from body_language.detect import classify
from small_lm import generate_text, apply_chat_template

def get_diar_segment_with_transcript(video_path):
    """
    Returns speaker-labeled transcripts with start and end times.
    """
    result, diarize_segment = get_diar_segment(video_path)

    # Consolidate segments by speaker
    consolidated = {}

    for segment in result:
        start = segment['start']
        end = segment['end']
        text = segment['text']

        # Find corresponding diarize segment
        matching_speakers = diarize_segment[(diarize_segment['start'] <= start) & (diarize_segment['end'] >= end)]

        if not matching_speakers.empty:
            speaker = matching_speakers.iloc[0]['speaker']
            
            if speaker not in consolidated:
                consolidated[speaker] = []
            
            consolidated[speaker].append({'text': text, 'start': start, 'end': end})
    return consolidated

if __name__ == "__main__":
    video_path = 'path to your directory'
    output_dir = 'path to output directory'
    spkr_label_transcript = get_diar_segment_with_transcript(video_path)

    # Combine text for each speaker
    combined_text = {speaker: ' '.join([seg['text'] for seg in segments]) for speaker, segments in spkr_label_transcript.items()}

    # Detect personality for each speaker
    personality_results = {speaker: personality_detection(text) for speaker, text in combined_text.items()}

    # Classify CV modality
    cv_results = classify(spkr_label_transcript, video_path, output_dir)

    # Combine results
    final_results = {speaker: {'personality': personality_results[speaker], 'cv_confidence': cv_results[speaker]} for speaker in personality_results}

    # Use small LM to generate a text explanation for each speaker
    for speaker, data in final_results.items():
        messages = [{"role": "user", "content": f"Explain the personality traits and CV confidence for speaker {speaker} with {data['personality']} and {data['cv_confidence'] * 100}% confidence."}]
        prompt = apply_chat_template(messages)
        generated_text = generate_text(prompt)
        final_results[speaker]['explanation'] = generated_text

    # Export final results to JSON
    output_json_path = os.path.join(output_dir, 'final_results.json')
    with open(output_json_path, 'w') as json_file:
        json.dump(final_results, json_file, indent=4)

    print(f"Final results saved to {output_json_path}")
