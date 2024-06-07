#this version y axis is flipped
import cv2
import mediapipe as mp
import numpy as np

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

cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

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
            right_wrist_coordinates = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width,
                                       results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * image_height]
            left_elbow_coordinates = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * image_width,
                                      results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * image_height]
            right_elbow_coordinates = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * image_width,
                                       results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * image_height]
            left_shoulder_coordinates = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width,
                                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height]
            right_shoulder_coordinates = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width,
                                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height]

            if results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].visibility < 0.5 and \
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].visibility < 0.5:
                visible_elbows = [left_elbow_coordinates, right_elbow_coordinates]
                visible_shoulders = [left_shoulder_coordinates, right_shoulder_coordinates]

                if len(visible_elbows) > 0 and len(visible_shoulders) > 0:
                    elbow_y_avg = sum(coord[1] for coord in visible_elbows) / len(visible_elbows)
                    shoulder_y_avg = sum(coord[1] for coord in visible_shoulders) / len(visible_shoulders)

                    if elbow_y_avg <= shoulder_y_avg+5:
                        confidence = True
            else:
                wrist_coordinates = [left_wrist_coordinates, right_wrist_coordinates]
                shoulder_coordinates = [left_shoulder_coordinates, right_shoulder_coordinates]

                if any(wrist[1] < shoulder[1] for wrist in wrist_coordinates for shoulder in shoulder_coordinates):
                    confidence = True
                else:
                    triangle_area = calculate_triangle_area(nose_coordinates, left_wrist_coordinates, right_wrist_coordinates)
                    quadrilateral_area = calculate_quadrilateral_area(left_shoulder_coordinates, right_shoulder_coordinates,
                                                                      left_elbow_coordinates, right_elbow_coordinates)

                    if triangle_area > quadrilateral_area:
                        confidence = True

        # Display confidence status on the image
        confidence_text = "Confidence: " + ("True" if confidence else "False")
        cv2.putText(image, confidence_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Flip the image horizontally for a selfie-view display.
        # Display the image without flipping
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()