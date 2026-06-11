import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tf.keras.models.load_model(
    "models/nail_model.h5"
)

val_gen = ImageDataGenerator(rescale=1./255)

val_data = val_gen.flow_from_directory(
    "data/val",
    target_size=(224,224),
    batch_size=16,
    class_mode="categorical",
    shuffle=False
)

predictions = model.predict(val_data)

y_pred = np.argmax(predictions, axis=1)
y_true = val_data.classes

class_names = list(val_data.class_indices.keys())

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)