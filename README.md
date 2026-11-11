# Eye Look Classifier (MobileNetV2 CNN)

A webcam-based educational project that classifies visible eye-look images with a MobileNetV2 transfer-learning CNN. It can train from any image dataset organized into one subfolder per class. For the supplied Eye dataset, the classes are `close_look`, `forward_look`, `left_look`, and `right_look`; the webcam app detects a face and classifies an eye crop.

> This predicts the folder label from pixels; it does not measure a person's mental attention. Accuracy depends on data quality and how closely training images match webcam images. It can be wrong with glasses, lighting, camera angle, multiple faces, or unusual posture. Do not use it for employee/student surveillance or high-stakes decisions.

## Setup (Python 3.10 or 3.11)
```bash
py -3.11 -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

The TensorFlow version in `requirements.txt` does not support the Python 3.14 runtime. Use a supported Python version such as 3.11.

## Step 1: Train from a labeled dataset

Put each label in its own folder, with images directly inside it:
```text
Eye dataset/
  close_look/
  forward_look/
  left_look/
  right_look/
```

Train using the dataset path (quote paths that contain spaces):
```bash
python train.py --data-dir "C:\Users\abdmo\Documents\Eye dataset" --epochs 12
```

Training streams images in batches rather than loading the full dataset into memory. It reserves 20% for validation, reports per-class metrics and a confusion matrix, and saves `models/focus_model.keras` plus `models/class_names.json`. The validation score is an estimate on a random image split, not a guarantee of webcam performance.

## Optional: Collect webcam examples

The collector creates face-crop classes for the original orientation-estimation workflow. These labels are not interchangeable with the eye-look labels above.
```bash
python collect_data.py
```
A webcam window opens. Press:
- `1` to save the current face crop as `focused`
- `2` to save as `partially_focused`
- `3` to save as `unfocused`
- `q` to quit

## Run live webcam inference
```bash
python app.py
```
Press `q` to quit. The overlay shows the predicted dataset class and confidence. For the four eye-look classes, the app uses OpenCV's Haar face and eye detectors to crop the eye before inference.

## Dataset options
Eye-look and gaze direction labels are not equivalent to mental attention. Public gaze datasets can help with gaze-direction research, but their labels are not equivalent to attention:
- GazeCapture (MIT): https://gazecapture.csail.mit.edu/
- MPIIGaze: https://www.perceptualui.org/research/datasets/MPIIGaze/
- RT-GENE: https://github.com/Tobias-Fischer/rt_gene

Use each dataset under its own access and license terms. Do not relabel gaze direction as mental focus without a defensible definition. Use the webcam collector only when training the separate face-orientation classes, and collect consented examples that match the deployment setting.

## Project structure
- `collect_data.py` — capture and label face crops
- `train.py` — MobileNetV2 transfer learning and evaluation
- `app.py` — real-time webcam prediction
- `src/` — model and preprocessing helpers
- `data/` — generated labeled images
- `models/` — trained CNN and class names
