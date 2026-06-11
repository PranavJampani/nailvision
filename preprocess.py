import cv2
import os

INPUT_DIR = "raw_images"
OUTPUT_DIR = "processed_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for img_name in os.listdir(INPUT_DIR):
    img_path = os.path.join(INPUT_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    img = cv2.resize(img, (224, 224))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), img)

print("Preprocessing complete.")
