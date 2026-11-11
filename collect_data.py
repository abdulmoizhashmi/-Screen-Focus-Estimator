from pathlib import Path
import cv2
from src.vision import face_detector, largest_face, crop_face, CLASSES

ROOT = Path("data")
KEYS = {"1": "focused", "2": "partially_focused", "3": "unfocused"}

def main():
    for name in CLASSES:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    detector = face_detector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check camera permissions or camera index.")
    print("Press 1=focused, 2=partially_focused, 3=unfocused to save face crop; q=quit.")
    counts = {name: len(list((ROOT/name).glob("*.jpg"))) for name in CLASSES}
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rect = largest_face(frame, detector)
            display = frame.copy()
            face = None
            if rect is not None:
                x,y,w,h = rect
                cv2.rectangle(display, (x,y), (x+w,y+h), (0,255,0), 2)
                face = crop_face(frame, rect)
            cv2.putText(display, "1 Focused | 2 Partial | 3 Unfocused | Q Quit",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0,255,255), 2)
            cv2.putText(display, f"Saved: {counts}", (12, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1)
            if face is None:
                cv2.putText(display, "No face detected", (12, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)
            cv2.imshow("Collect Focus Training Images", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if chr(key) in KEYS and face is not None:
                label = KEYS[chr(key)]
                counts[label] += 1
                path = ROOT / label / f"{label}_{counts[label]:05d}.jpg"
                # OpenCV expects BGR when writing.
                cv2.imwrite(str(path), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                print("Saved", path)
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
