import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from preprocess import get_datasets, CLASS_NAMES

SHORT_NAMES = [
    "Bacterial Spot", "Early Blight", "Healthy", "Late Blight",
    "Leaf Mold", "Septoria", "Spider Mites", "Target Spot",
    "Mosaic Virus", "Yellow Leaf Curl"
]

def evaluate():
    print("Loading model...")
    model = tf.keras.models.load_model("models/model.h5")

    print("Loading test dataset...")
    _, test_ds = get_datasets(
        train_dir="data/train",
        test_dir="data/test"
    )

    print("Running predictions...")
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=SHORT_NAMES))

    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png")
    print("Confusion matrix saved to models/confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    evaluate()