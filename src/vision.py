from pathlib import Path
import cv2
import numpy as np

IMG_SIZE = (160, 160)
CLASSES = ["focused", "partially_focused", "unfocused"]

def face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("Could not load OpenCV Haar face detector.")
    return detector


def eye_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("Could not load OpenCV Haar eye detector.")
    return detector


def largest_face(frame, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(70, 70))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda r: r[2] * r[3])


def largest_eye(face, detector):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    eyes = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
    )
    if len(eyes) == 0:
        return None
    return max(eyes, key=lambda r: r[2] * r[3])


def crop_face(frame, rect, margin=0.18):
    x, y, w, h = rect
    mx, my = int(w * margin), int(h * margin)
    x1, y1 = max(0, x-mx), max(0, y-my)
    x2, y2 = min(frame.shape[1], x+w+mx), min(frame.shape[0], y+h+my)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), IMG_SIZE)


def crop_eye(face, rect, margin=0.35):
    x, y, w, h = rect
    mx, my = int(w * margin), int(h * margin)
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2, y2 = min(face.shape[1], x + w + mx), min(face.shape[0], y + h + my)
    crop = face[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), IMG_SIZE)


def prepare_for_model(rgb_face):
    arr = np.asarray(rgb_face, dtype=np.float32)
    return np.expand_dims(arr, axis=0)
