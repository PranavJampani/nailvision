import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load model
model = tf.keras.models.load_model("models/nail_model.h5")

# Validation data
val_gen = ImageDataGenerator(rescale=1./255)

val_data = val_gen.flow_from_directory(
    "data/val",
    target_size=(224,224),
    batch_size=16,
    class_mode="categorical",
    shuffle=False
)

# Predictions
predictions = model.predict(val_data)

y_pred = np.argmax(predictions, axis=1)
y_true = val_data.classes

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# Labels
class_names = list(val_data.class_indices.keys())

# Plot
plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()