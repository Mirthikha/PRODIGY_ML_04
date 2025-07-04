from preprocess import load_data
from model import build_model
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
import os

data_path = "data"  

X, y, label_encoder = load_data(data_path)


print("Data loaded successfully!")
print(f"Number of samples: {X.shape[0]}")
print(f"Input shape: {X.shape[1:]}")
print(f"Number of classes: {len(label_encoder)}")
print("Label Map:", label_encoder)


model = build_model(input_shape=(64, 64, 1), num_classes=len(label_encoder))


model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2,
          callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])


os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/gesture_model.h5")

print("Model trained and saved successfully.")
