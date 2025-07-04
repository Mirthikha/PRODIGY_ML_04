

import cv2
import numpy as np
from tensorflow.keras.models import load_model

def predict_image(image_path, model_path, label_map, img_size=(64, 64)):
    model = load_model(model_path)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: Could not read image '{image_path}'")
        return

    img = cv2.resize(img, img_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)  # add channel dimension: (64,64,1)
    img = np.expand_dims(img, axis=0)   # batch dimension: (1,64,64,1)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    print(f"Predicted Gesture: {label_map[predicted_class]}")
    print(f"Confidence: {confidence:.2f}")
