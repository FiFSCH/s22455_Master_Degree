import mediapipe as mp
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

smoothing_factor = 0.8
previous_landmarks = None
previous_bbox = None

landmark_buffer = deque(maxlen=5)
bbox_buffer = deque(maxlen=5)


def smooth_landmarks(landmarks):
    global previous_landmarks
    if previous_landmarks is None:
        previous_landmarks = landmarks
        return landmarks

    smoothed_landmarks = []
    for prev, curr in zip(previous_landmarks, landmarks):
        x = smoothing_factor * prev.x + (1 - smoothing_factor) * curr.x
        y = smoothing_factor * prev.y + (1 - smoothing_factor) * curr.y
        z = smoothing_factor * prev.z + (1 - smoothing_factor) * curr.z
        smoothed_landmarks.append(type(curr)(x=x, y=y, z=z))

    previous_landmarks = smoothed_landmarks
    return smoothed_landmarks


def average_landmarks(new_landmarks):
    landmark_buffer.append(new_landmarks)
    averaged_landmarks = []
    for values in zip(*landmark_buffer):
        x = sum([v.x for v in values]) / len(values)
        y = sum([v.y for v in values]) / len(values)
        z = sum([v.z for v in values]) / len(values)
        averaged_landmarks.append(type(values[0])(x=x, y=y, z=z))
    return averaged_landmarks


def smooth_bounding_box(current_bbox):
    global previous_bbox
    if previous_bbox is None:
        previous_bbox = current_bbox
        return current_bbox

    smoothed_bbox = [int(smoothing_factor * prev + (1 - smoothing_factor) * curr) for prev, curr in zip(previous_bbox, current_bbox)]
    previous_bbox = smoothed_bbox
    return smoothed_bbox


def average_bbox(new_bbox):
    bbox_buffer.append(new_bbox)
    averaged_bbox = [int(sum(c) / len(c)) for c in zip(*bbox_buffer)]
    return averaged_bbox


def is_outlier(current_bbox, threshold=50):
    global previous_bbox
    if previous_bbox is None:
        return False
    return any(abs(c1 - c2) > threshold for c1, c2 in zip(current_bbox, previous_bbox))