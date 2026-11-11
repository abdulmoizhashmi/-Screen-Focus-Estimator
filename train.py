import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.vision import IMG_SIZE


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def load_data(data_dir, batch_size, seed):
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_path}")

    class_names = sorted(path.name for path in data_path.iterdir() if path.is_dir())
    if len(class_names) < 2:
        raise RuntimeError(
            f"Expected at least two class subfolders in {data_path}; found {class_names}"
        )

    image_groups = defaultdict(list)
    for label, name in enumerate(class_names):
        for path in sorted((data_path / name).iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                image_groups[sha256(path.read_bytes()).digest()].append((label, path))

    image_paths, labels = [], []
    duplicate_files = 0
    conflicting_groups = 0
    for group in image_groups.values():
        group_labels = {label for label, _ in group}
        if len(group_labels) > 1:
            conflicting_groups += 1
            continue
        duplicate_files += len(group) - 1
        label, path = group[0]
        image_paths.append(str(path))
        labels.append(label)

    counts = np.bincount(labels, minlength=len(class_names))
    count_by_class = dict(zip(class_names, counts.tolist()))
    if np.any(counts < 2):
        raise RuntimeError(
            "Need at least 2 unique, unambiguous images in each class after "
            f"deduplication. Counts: {count_by_class}"
        )

    print(f"Removed {duplicate_files} exact duplicate files.")
    print(f"Excluded {conflicting_groups} duplicate groups with conflicting labels.")
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths,
        np.asarray(labels, dtype=np.int32),
        test_size=0.20,
        random_state=seed,
        stratify=labels,
    )
    train_counts = np.bincount(train_labels, minlength=len(class_names))
    class_weights = {
        index: float(len(train_labels) / (len(class_names) * count))
        for index, count in enumerate(train_counts)
    }

    def make_dataset(paths, target_labels, shuffle):
        dataset = tf.data.Dataset.from_tensor_slices((paths, target_labels))

        def decode_image(path, label):
            image = tf.io.decode_image(
                tf.io.read_file(path), channels=3, expand_animations=False
            )
            image.set_shape([None, None, 3])
            return tf.image.resize(image, IMG_SIZE), label

        if shuffle:
            dataset = dataset.shuffle(
                len(paths), seed=seed, reshuffle_each_iteration=True
            )
        return (
            dataset.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

    return (
        make_dataset(train_paths, train_labels, shuffle=True),
        make_dataset(val_paths, val_labels, shuffle=False),
        class_names,
        count_by_class,
        class_weights,
    )


def build_model(num_classes):
    augmentation = tf.keras.Sequential(
        [
            layers.RandomRotation(0.03),
            layers.RandomZoom(0.05),
            layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


def main():
    parser = argparse.ArgumentParser(
        description="Train an eye-look classifier from class-named image folders."
    )
    parser.add_argument("--data-dir", default="data", help="Folder containing one subfolder per class.")
    parser.add_argument("--output-dir", default="models", help="Where to save the model and labels.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    tf.keras.utils.set_random_seed(args.seed)
    train_ds, val_ds, class_names, counts, class_weights = load_data(
        args.data_dir, args.batch_size, args.seed
    )
    print(f"Classes: {class_names}")
    print(f"Image counts: {counts}")
    print(f"Training class weights: {class_weights}")

    model, base = build_model(len(class_names))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=4, restore_best_weights=True, monitor="val_loss"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=2, factor=0.3, monitor="val_loss"
        ),
    ]

    print("Training classifier head...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    base.trainable = True
    for layer in base.layers[:-25]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("Fine-tuning the final MobileNetV2 layers...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max(3, args.epochs // 2),
        callbacks=callbacks,
        class_weight=class_weights,
    )

    y_true = np.concatenate([labels.numpy() for _, labels in val_ds])
    probabilities = model.predict(val_ds, verbose=0)
    y_pred = probabilities.argmax(axis=1)
    print(
        "\nValidation report:\n",
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        ),
    )
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "focus_model.keras")
    (output_dir / "class_names.json").write_text(
        json.dumps(class_names, indent=2), encoding="utf-8"
    )
    print(f"Saved model and labels to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
