import os
import cv2
import numpy as np

def load_data(data_path):
    X = []
    y = []
    label_map = {}
    label_counter = 0
    img_size = (64, 64)

    # Go through each person (00 to 09)
    for person in os.listdir(data_path):
        person_path = os.path.join(data_path, person)

        if not os.path.isdir(person_path):
            continue  # skip if not a folder

        for gesture in os.listdir(person_path):
            gesture_path = os.path.join(person_path, gesture)

            if not os.path.isdir(gesture_path):
                continue  # skip if not a folder

            # Assign a unique label for each gesture
            if gesture not in label_map:
                label_map[gesture] = label_counter
                label_counter += 1

            for img_file in os.listdir(gesture_path):
                img_path = os.path.join(gesture_path, img_file)

                # Load image
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    print(f"[⚠️ Skipped] Cannot read: {img_path}")
                    continue

                img = cv2.resize(img, img_size)
                X.append(img)
                y.append(label_map[gesture])

    X = np.array(X).reshape(-1, 64, 64, 1) / 255.0
    y = np.array(y)

    return X, y, label_map
