import json
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from src.vision import (
    crop_face,
    crop_eye,
    face_detector,
    eye_detector,
    largest_eye,
    largest_face,
    prepare_for_model,
)

MODEL_PATH = Path("models/focus_model.keras")
LABEL_PATH = Path("models/class_names.json")
EYE_LOOK_CLASSES = {"close_look", "forward_look", "left_look", "right_look"}


def main():
    if not MODEL_PATH.exists() or not LABEL_PATH.exists():
        print("Model missing. Train a model first; see the training instructions in README.md.")
        return
    model = tf.keras.models.load_model(MODEL_PATH)
    classes = json.loads(LABEL_PATH.read_text(encoding="utf-8"))
    use_eye_crops = set(classes) == EYE_LOOK_CLASSES
    face_cascade = face_detector()
    eye_cascade = eye_detector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check camera permissions or camera index.")
    print("Live classifier started. Press q in the video window to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            face_rect = largest_face(frame, face_cascade)
            if face_rect is None:
                cv2.putText(frame, "No face detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            else:
                x, y, w, h = face_rect
                face = frame[y:y+h, x:x+w]
                image = None
                box = face_rect
                if use_eye_crops:
                    eye_rect = largest_eye(face, eye_cascade)
                    if eye_rect is None:
                        cv2.putText(frame, "No eye detected", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                    else:
                        ex, ey, ew, eh = eye_rect
                        image = crop_eye(face, eye_rect)
                        box = (x+ex, y+ey, ew, eh)
                else:
                    image = crop_face(frame, face_rect)
                    box = face_rect
                if image is not None:
                    probs = model.predict(prepare_for_model(image), verbose=0)[0]
                    idx = int(np.argmax(probs))
                    label = classes[idx].replace("_", " ").title()
                    confidence = float(probs[idx])
                    color = (0, 200, 255)
                    bx, by, bw, bh = box
                    cv2.rectangle(frame, (bx,by), (bx+bw,by+bh), color, 2)
                    cv2.putText(frame, f"{label}: {confidence:.0%}",
                                (bx, max(28,by-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            status = "Eye-look category estimate" if use_eye_crops else "Orientation estimate"
            cv2.putText(frame, f"{status} only | Q: quit", (12, frame.shape[0]-18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.imshow("Screen Focus Estimator", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
