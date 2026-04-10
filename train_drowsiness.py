import tensorflow as tf
from tensorflow.keras import layers, models

# Settings
img_size = (64, 64)
batch_size = 32

# Load dataset
train = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/train",
    image_size=img_size,
    batch_size=batch_size
)

val = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/val",
    image_size=img_size,
    batch_size=batch_size
)

# Normalize
normalization_layer = layers.Rescaling(1./255)

train = train.map(lambda x, y: (normalization_layer(x), y))
val = val.map(lambda x, y: (normalization_layer(x), y))

# Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(train, validation_data=val, epochs=5)

# Save model
model.save("models/drowsiness_model.h5")

print("Model saved successfully")