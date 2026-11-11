import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import tensorflow as tf


IMG_SIZE = (160, 160)


def make_inference_model(model):
    base = next(
        (
            layer
            for layer in model.layers
            if isinstance(layer, tf.keras.Model)
            and not isinstance(layer, tf.keras.Sequential)
        ),
        None,
    )
    pooling = next(
        (
            layer
            for layer in model.layers
            if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D)
        ),
        None,
    )
    classifier = next(
        (
            layer
            for layer in reversed(model.layers)
            if isinstance(layer, tf.keras.layers.Dense)
        ),
        None,
    )
    if base is None or pooling is None or classifier is None:
        raise RuntimeError(
            "The trained model must contain a MobileNetV2 base, global average "
            "pooling, and a dense classifier."
        )

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3), name="images")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = pooling(x)
    outputs = classifier(x)
    return tf.keras.Model(inputs, outputs, name="eye_look_inference")


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
    inference_model = make_inference_model(model)
    if inference_model.output_shape[-1] != len(class_names):
        raise ValueError(
            f"Model output has {inference_model.output_shape[-1]} classes, "
            f"but {len(class_names)} class names were provided."
        )

    sample = tf.zeros([1, *IMG_SIZE, 3], dtype=tf.float32)
    tf.debugging.assert_near(
        model(sample, training=False),
        inference_model(sample, training=False),
        message="Inference-only model does not match the trained model output.",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="eye-look-saved-model-") as temp_dir:
        saved_model_dir = Path(temp_dir) / "saved_model"

        @tf.function(
            input_signature=[
                tf.TensorSpec([None, *IMG_SIZE, 3], tf.float32, name="images")
            ]
        )
        def serve(images):
            return {"probabilities": inference_model(images, training=False)}

        tf.saved_model.save(
            inference_model,
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
