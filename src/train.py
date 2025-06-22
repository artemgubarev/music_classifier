import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
from tqdm import tqdm
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split

genres = {
    0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop',
    5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'
}

def extract_features(audio_path, n_mfcc=13, sr=22050, n_fft=2048,
                     hop_length=512, duration=30, num_segments=10):
    return np.array(segments_mfcc) if segments_mfcc else None

X, y = [], []
BASE_PATH = "gtzan_data/Data/genres_original"
for label, genre in genres.items():
    for i in tqdm(range(100), desc=f"Processing {genre}"):
        if features is not None:
            X.extend(features)
            y.extend([label] * len(features))

X = np.array(X)[..., np.newaxis]
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

def build_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
        BatchNormalization(),
        Conv2D(32, (2, 2), activation='relu'),
        MaxPooling2D((2, 2), strides=(2, 2), padding='same'),
        BatchNormalization(),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(10, activation='softmax')
    ])
    return model

model = build_model(input_shape=(X.shape[1], X.shape[2], 1))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=100)

def plot_history(history):
    plt.show()

plot_history(history)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc:.4f}')
model.save("genre_classification_cnn.h5")