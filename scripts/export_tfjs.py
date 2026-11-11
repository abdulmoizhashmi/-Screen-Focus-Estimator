import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import tensorflow as tf


IMG_SIZE = (160, 160)


def main():
    parser = argparse.ArgumentParser(
        description="Export the trained Keras classifier for browser inference."
    )
    parser.add_argument("--model", default="models/focus_model.keras")
    parser.add_argument("--classes", default="models/class_names.json")
    parser.add_argument("--output-dir", default="web/public/model")
    args = parser.parse_args()

    model_path = Path(args.model)
    class_path = Path(args.classes)
    output_dir = Path(args.output_dir)
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    if not class_path.is_file():
        raise FileNotFoundError(f"Class names not found: {class_path}")

    with class_path.open(encoding="utf-8") as file:
        class_names = json.load(file)
    if not isinstance(class_names, list) or not class_names:
        raise ValueError(f"Expected a non-empty JSON list of class names in {class_path}")

    converter = shutil.which("tensorflowjs_converter")
    if converter is None:
        raise RuntimeError(
            "tensorflowjs_converter is missing. Install the tensorflowjs package first."
        )

    model = tf.keras.models.load_model(model_path, compile=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="eye-look-saved-model-") as temp_dir:
        saved_model_dir = Path(temp_dir) / "saved_model"

        @tf.function(
            input_signature=[
                tf.TensorSpec([None, *IMG_SIZE, 3], tf.float32, name="images")
            ]
        )
        def serve(images):
            return {"probabilities": model(images, training=False)}

        tf.saved_model.save(
            model,
            str(saved_model_dir),
            signatures={"serving_default": serve},
        )
        subprocess.run(
            [
                converter,
                "--input_format=tf_saved_model",
                "--saved_model_tags=serve",
                "--signature_name=serving_default",
                str(saved_model_dir),
                str(output_dir),
            ],
            check=True,
        )

    model_json = output_dir / "model.json"
    if not model_json.is_file():
        raise RuntimeError(f"TensorFlow.js conversion did not create {model_json}")
    (output_dir / "class_names.json").write_text(
        json.dumps(class_names, indent=2), encoding="utf-8"
    )
    print(f"Exported browser model and labels to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
