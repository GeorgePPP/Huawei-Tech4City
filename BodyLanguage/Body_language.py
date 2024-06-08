import cv2
import mediapipe as mp
import pandas as pd
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def calculate_triangle_area(point1, point2, point3):
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2)

def calculate_quadrilateral_area(point1, point2, point3, point4):
    triangle1_area = calculate_triangle_area(point1, point2, point3)
    triangle2_area = calculate_triangle_area(point1, point3, point4)
    return triangle1_area + triangle2_area

def format_time(time_in_seconds):
    hours = int(time_in_seconds // 3600)
    minutes = int((time_in_seconds % 3600) // 60)
    seconds = int(time_in_seconds % 60)
    milliseconds = int((time_in_seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

cap = cv2.VideoCapture('path/to/your/videos')

if not cap.isOpened():
    print("Error opening the video file.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

confidence_intervals = []
current_confidence = False
start_time = 0
end_time = 0
segment_count = 0

output_dir = 'path/to/your/output'
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, 'output_video.mp4')
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_height, image_width, _ = image.shape
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        confidence = False
        if results.pose_landmarks:
            nose_coordinates = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width,
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height]
            left_wrist_coordinates = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * image_width,
                                      results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * image_height]
            right_wrist_coordinates = [
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width,
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * image_height]
            left_elbow_coordinates = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * image_width,
                                      results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * image_height]
            right_elbow_coordinates = [
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * image_width,
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * image_height]
            left_shoulder_coordinates = [
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width,
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height]
            right_shoulder_coordinates = [
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width,
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height]

            if results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].visibility < 0.5 and \
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].visibility < 0.5:
                visible_elbows = [left_elbow_coordinates, right_elbow_coordinates]
                visible_shoulders = [left_shoulder_coordinates, right_shoulder_coordinates]

                if len(visible_elbows) > 0 and len(visible_shoulders) > 0:
                    elbow_y_avg = sum(coord[1] for coord in visible_elbows) / len(visible_elbows)
                    shoulder_y_avg = sum(coord[1] for coord in visible_shoulders) / len(visible_shoulders)

                    if elbow_y_avg <= shoulder_y_avg + 5:
                        confidence = True
            else:
                wrist_coordinates = [left_wrist_coordinates, right_wrist_coordinates]
                shoulder_coordinates = [left_shoulder_coordinates, right_shoulder_coordinates]
                elbow_coordinates = [left_elbow_coordinates, right_elbow_coordinates]
                if any(wrist[1] < shoulder[1] for wrist in wrist_coordinates for shoulder in shoulder_coordinates):
                    confidence = True
                else:
                    triangle_area = calculate_triangle_area(nose_coordinates, left_wrist_coordinates,
                                                            right_wrist_coordinates)
                    quadrilateral_area = calculate_quadrilateral_area(left_shoulder_coordinates,
                                                                      right_shoulder_coordinates,
                                                                      left_elbow_coordinates, right_elbow_coordinates)
                    shoulder_triangle = calculate_triangle_area(nose_coordinates, left_shoulder_coordinates,
                                                                right_shoulder_coordinates)
                    if any(wrist[1] < elbow[1] for wrist in wrist_coordinates for elbow in elbow_coordinates):
                        elbow_triangle = calculate_triangle_area(nose_coordinates, left_elbow_coordinates,
                                                                 right_elbow_coordinates)
                        if elbow_triangle > quadrilateral_area:
                            confidence = True
                    elif (shoulder_triangle + triangle_area) > quadrilateral_area:
                        confidence = True

        if confidence != current_confidence:
            if current_confidence is not None:
                end_time = current_time
                formatted_start_time = format_time(start_time)
                formatted_end_time = format_time(end_time)
                confidence_intervals.append({'segment': f"[{formatted_start_time} --> {formatted_end_time}]", 'start': start_time, 'end': end_time, 'label': str(current_confidence)})
                segment_count += 1
            start_time = current_time
            current_confidence = confidence

        if confidence:
            confidence_text = "Confidence: True"
        else:
            confidence_text = "Confidence: False"
        cv2.putText(image, confidence_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(image)

        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) == ord('q'):
            break

    end_time = current_time
    formatted_start_time = format_time(start_time)
    formatted_end_time = format_time(end_time)
    confidence_intervals.append({'segment': f"[{formatted_start_time} --> {formatted_end_time}]", 'start': start_time, 'end': end_time, 'label': str(current_confidence)})

cap.release()
out.release()
cv2.destroyAllWindows()

# Display confidence intervals as a DataFrame
df = pd.DataFrame(confidence_intervals)
df.columns = ['Segment', 'Start', 'End', 'Confidence']
print(df)

# Save confidence intervals as an Excel file
excel_path = os.path.join(output_dir, 'confidence_intervals.xlsx')
df.to_excel(excel_path, index=False)