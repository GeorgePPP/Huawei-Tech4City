from flask import Flask, request, jsonify
from functions import generate_report, analyze_meeting, improve_body_language, generate_meeting_report_from_labels

app = Flask(__name__)

# Automatic Speech Recognition + Translation + Speaker Diarization
@app.route('/asr_translation_diarization', methods=['POST'])
def asr_translation_diarization():
    speech_input = request.files['audio']
    # Your function call here
    result = generate_report.predict(speech_input)
    return jsonify(result)

# Meeting Analysis (Speaker Diarization + NLP)
@app.route('/meeting_analysis', methods=['POST'])
def meeting_analysis():
    transcribed_file = request.files['transcription']
    # Your function call here
    result = analyze_meeting(transcribed_file)
    return jsonify(result)

# Body Language Improvement (CV Body Keypoints Detection)
@app.route('/body_language_improvement', methods=['POST'])
def body_language_improvement():
    image_feed = request.files['image']
    # Your function call here
    result = improve_body_language(image_feed)
    return jsonify(result)

# Large Language Model (LLM)
@app.route('/generate_meeting_report', methods=['POST'])
def generate_meeting_report():
    nlp_labels = request.json['labels']
    # Your function call here
    result = generate_meeting_report_from_labels(nlp_labels)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
