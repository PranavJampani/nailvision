import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model(
    "models/nail_model.h5"
)

IMG_PATH = "test.jpg"

img = cv2.imread(IMG_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_resized = cv2.resize(img, (224,224))

img_array = img_resized / 255.0
img_array = np.expand_dims(img_array, axis=0)

last_conv_layer = None

for layer in reversed(model.layers):
    if len(layer.output.shape) == 4:
        last_conv_layer = layer.name
        break

print("Using layer:", last_conv_layer)

grad_model = tf.keras.models.Model(
    [model.inputs],
    [
        model.get_layer(last_conv_layer).output,
        model.output
    ]
)

with tf.GradientTape() as tape:

    conv_outputs, predictions = grad_model(img_array)

    class_idx = tf.argmax(predictions[0])

    loss = predictions[:, class_idx]

grads = tape.gradient(loss, conv_outputs)

pooled_grads = tf.reduce_mean(
    grads,
    axis=(0,1,2)
)

conv_outputs = conv_outputs[0]

heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

heatmap = tf.squeeze(heatmap)

heatmap = np.maximum(
    heatmap,
    0
)

heatmap /= np.max(heatmap)

heatmap = cv2.resize(
    heatmap,
    (224,224)
)

heatmap = np.uint8(255 * heatmap)

heatmap = cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_JET
)

overlay = cv2.addWeighted(
    img_resized,
    0.6,
    heatmap,
    0.4,
    0
)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img_resized)
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(overlay)
plt.title("Grad-CAM")
plt.axis("off")

plt.show()