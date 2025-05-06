import mediapipe as mp
import cv2
import torch
from PIL import Image
from Auxilary.Live_camera_footage_helpers import smooth_landmarks, average_landmarks, smooth_bounding_box, average_bbox, is_outlier


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


def capture_camera_footage(transform, model, class_names):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(image=frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = smooth_landmarks(hand_landmarks.landmark)
                landmarks = average_landmarks(landmarks)

                # Get bbox
                min_x = min([l.x for l in landmarks]) * frame.shape[1]
                max_x = max([l.x for l in landmarks]) * frame.shape[1]
                min_y = min([l.y for l in landmarks]) * frame.shape[0]
                max_y = max([l.y for l in landmarks]) * frame.shape[0]

                margin = int(0.1 * (max_x - min_x))
                x1 = max(0, int(min_x - margin))
                y1 = max(0, int(min_y - margin))
                x2 = min(frame.shape[1], int(max_x + margin))
                y2 = min(frame.shape[0], int(max_y + margin))

                bbox = smooth_bounding_box([x1, y1, x2, y2])
                bbox = average_bbox(bbox)

                if is_outlier(bbox):
                    continue

                # Get hand
                try:
                    hand_region = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    hand_region_pil = Image.fromarray(cv2.cvtColor(hand_region, cv2.COLOR_BGR2RGB))
                    hand_tensor = transform(hand_region_pil).unsqueeze(0)
                except Exception as e:
                    print(f"{e}")
                    continue

                # Make prediction
                with torch.no_grad():
                    output = model(hand_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    class_name = class_names[predicted.item()]
                    confidence_level = confidence.item()

                # Draw bbox and prediction
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                text = f"{class_name} ({confidence_level:.2f})"
                cv2.putText(frame, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # Smooth and display landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3),  # landmarks
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3)  # connections between landmarks
                )

        # Show GUI
        cv2.imshow('Hand Gesture Recognition', frame)

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


