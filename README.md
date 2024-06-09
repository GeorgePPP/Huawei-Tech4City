# EmpowerYou - Huawei Tech4City Competition

## Overview
This project aims to develop a system for generating meeting analysis for meeting attendees using multimodal inputs. The system extracts features from speech and images to classify personal traits, ultimately generating attendee-specific reports.

## Input Types
- Speech Signal
- Image

## Workflow
### First Stage: Multimodal Feature Extraction
1. **Automatic Speaker Recognition:** Processes the speech signal to identify and recognize the speaker.
2. **Body Keypoints Detection:** Analyzes images to detect key points on the body, which are critical for understanding body language.
3. **Speaker Diarization:** Separates and identifies speech segments according to different speakers within the meeting context.

### Second Stage: Personal Traits Analysis
1. **BERT-based Personal Traits Classification:** Uses a BERT model to classify personal traits based on the recognized speaker's voice.
2. **Body Language Classification:** Utilizes the detected body keypoints to classify the body language, providing insights into non-verbal cues.

### Last Stage: Report Generation
1. **Small Language Model:** Integrates the personal traits information derived from speech and body language analyses to generate a comprehensive report.
2. **Meeting Attendee-specific Report:** Compiles all analyzed data into detailed, attendee-specific reports which can be used for further review and analysis.

## Outcome
The system provides detailed, individualized reports for each meeting attendee, offering insights based on multimodal analysis of their speech and body language. This aids in understanding personal traits and behaviors during meetings, facilitating better communication and interaction strategies.

## Usage
1. **Data Input:** Provide speech signals and images of meeting attendees.
2. **Run Workflow:** Execute the provided code to process multimodal inputs, analyze personal traits, and generate reports.
3. **Review Reports:** Access the generated attendee-specific reports for insights into meeting dynamics and attendee behaviors.

## Dependencies
- Find in requirements.txt

## Contributors
- Yan Jinjiang, Ananya Varshney, George Ong, Cheo Le Xian, Ninad Dixit, Wong Yen Heng

## Notes
- This repository mainly shows the lowest hanging fruit of the development, it is far from optimal. If the team gets into semi-final round, this repository will be improved and scaled to production level.