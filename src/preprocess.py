import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
]

def get_datasets(train_dir="data/train", test_dir="data/test"):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
        class_names=CLASS_NAMES,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        class_names=CLASS_NAMES,
    )

    # Augmentation for training only
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomBrightness(0.2),
    ])

    # Normalize pixel values to [0, 1]
    normalization = tf.keras.layers.Rescaling(1.0 / 255)

    train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y))
    train_ds = train_ds.map(lambda x, y: (normalization(x), y))
    test_ds  = test_ds.map(lambda x, y: (normalization(x), y))

    # Prefetch for performance
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    test_ds  = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds