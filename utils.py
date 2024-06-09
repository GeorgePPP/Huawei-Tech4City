from whisper.whisper import get_diar_segment
from bert_personality.detect import personality_detection
    
def get_diar_segment_with_transcript(video_path):
    '''
    Return exp
    {'SPEAKER_02': [{'text': ' Bella.', 'start': 0.165, 'end': 1.065}, {'text': 'Gloria.', 'start': 1.326, 'end': 2.166}], 
    'SPEAKER_00': [{'text': 'I said she could stay with us, Marge, until she feels better.', 'start': 8.089, 'end': 10.15}]}
    '''
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
    video_path = r'C:\Users\User\Desktop\Huawei-Tech4City\whisper\sample.mp4'
    # spkr_label_transcript = get_diar_segment_with_transcript(video_path)

    spkr_label_transcript = {'SPEAKER_02': [{'text': ' Bella.', 'start': 0.165, 'end': 1.065}, 
                                            {'text': 'Gloria.', 'start': 1.326, 'end': 2.166}, 
                                            {'text': 'Love.', 'start': 2.506, 'end': 3.547}, 
                                            {'text': "Oh, I'm okay.", 'start': 5.348, 'end': 6.488}, 
                                            {'text': 'I will be.', 'start': 7.209, 'end': 7.809}, 
                                            {'text': "No, this won't be for long.", 'start': 12.271, 'end': 13.932}], 
                            'SPEAKER_00': [{'text': 'I said she could stay with us, Marge, until she feels better.', 'start': 8.089, 'end': 10.15}]}

    # Combine text for each speaker
    combined_text = {speaker: ' '.join([seg['text'] for seg in segments]) for speaker, segments in spkr_label_transcript.items()}

    # Detect personality for each speaker
    personality_results = {speaker: personality_detection(text) for speaker, text in combined_text.items()}

    # Output the results
    print(personality_results)

