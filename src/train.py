import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Make sure we always run from the project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from preprocess import get_datasets, CLASS_NAMES

# ── Config ──────────────────────────────────────────────
EPOCHS      = 15
MODEL_PATH  = "models/model.h5"
IMG_SIZE    = (224, 224)
NUM_CLASSES = 10
# ────────────────────────────────────────────────────────

def build_model():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # Freeze pretrained layers

    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compute_class_weights(train_dir="data/train"):
    counts = []
    for cls in CLASS_NAMES:
        folder = os.path.join(train_dir, cls)
        counts.append(len(os.listdir(folder)))

    total = sum(counts)
    weights = {}
    for i, count in enumerate(counts):
        weights[i] = total / (NUM_CLASSES * count)
    return weights


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"],     label="Train accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val accuracy")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"],     label="Train loss")
    axes[1].plot(history.history["val_loss"], label="Val loss")
    axes[1].set_title("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("models/training_history.png")
    print("Training plot saved to models/training_history.png")


def main():
    print("Loading datasets...")
    train_ds, test_ds = get_datasets()

    print("Computing class weights...")
    class_weights = compute_class_weights()
    print("Class weights:", class_weights)

    print("Building model...")
    model = build_model()
    model.summary()

    os.makedirs("models", exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, save_best_only=True, monitor="val_accuracy"
        ),
    ]

    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    plot_history(history)

    loss, accuracy = model.evaluate(test_ds)
    print(f"\nFinal test accuracy: {accuracy * 100:.2f}%")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()