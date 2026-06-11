import tensorflow as tf
import cv2
import numpy as np

class_names = [
    "Acral_Lentiginous_Melanoma",
    "Onychogryphosis",
    "blue_finger",
    "clubbing",
    "healthy",
    "pitting"
]

model = tf.keras.models.load_model(
    "models/nail_model.h5"
)

img_path = "test.jpg"

img = cv2.imread(img_path)

img = cv2.resize(
    img,
    (224,224)
)

img = img / 255.0

img = np.expand_dims(
    img,
    axis=0
)

predictions = model.predict(img)[0]

predicted_class = np.argmax(predictions)

print(
    "\nPrediction:",
    class_names[predicted_class]
)

print(
    "Confidence:",
    round(
        predictions[predicted_class] * 100,
        2
    ),
    "%"
)

print("\nTop Predictions:\n")

for i in np.argsort(predictions)[::-1]:
    print(
        f"{class_names[i]}: "
        f"{predictions[i]*100:.2f}%"
    )