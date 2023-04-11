import os
import numpy as np
import cv2
from keras.layers import Dense, Conv3D, MaxPooling3D, Dropout, Flatten, LSTM
from keras.models import Sequential
from keras.optimizers import Adam

# Set the paths to the violence and non-violence video clip folders
violence_path = "/dataset/NonViolence"
non_violence_path = "/dataset/Violence"

# Define the dimensions of the video frames
width = 224
height = 224

# Define the number of frames to sample from each video clip
num_frames = 20

# Define the number of classes
num_classes = 2

# Define the batch size and number of epochs
batch_size = 32
num_epochs = 10

# Define the RNN-CNN model
model = Sequential()

model.add(Conv3D(32, (3,3,3), activation='relu', input_shape=(num_frames, width, height, 3)))
model.add(MaxPooling3D(pool_size=(1,2,2)))
model.add(Dropout(0.25))

model.add(Conv3D(64, (3,3,3), activation='relu'))
model.add(MaxPooling3D(pool_size=(1,2,2)))
model.add(Dropout(0.25))

model.add(Conv3D(128, (3,3,3), activation='relu'))
model.add(MaxPooling3D(pool_size=(1,2,2)))
model.add(Dropout(0.25))

model.add(Conv3D(256, (3,3,3), activation='relu'))
model.add(MaxPooling3D(pool_size=(1,2,2)))
model.add(Dropout(0.25))

x = model.add(Flatten())
x=
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(128))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Define a function to extract frames from the video clips
def extract_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (width, height))
            frames.append(frame)
    cap.release()
    return np.array(frames)

# Load the violence video clips and extract the frames
violence_clips = os.listdir(violence_path)
X_violence = np.array([extract_frames(os.path.join(violence_path, clip)) for clip in violence_clips])

# Load the non-violence video clips and extract the frames
non_violence_clips = os.listdir(non_violence_path)
X_non_violence = np.array([extract_frames(os.path.join(non_violence_path, clip)) for clip in non_violence_clips])

# Combine the violence and non-violence data into a single dataset
X = np.concatenate((X_violence, X_non_violence))

# Create the labels for the data
y = np.concatenate((np.ones(len(X_violence)), np.zeros(len(X_non_violence))))

idxs = np.arange(len(X))
np.random.shuffle(idxs)
X = X[idxs]
y = y[idxs]

split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_val, y_val = X[split:], y[split:]


y_train = np.eye(num_classes)[y_train.astype('int')]
y_val = np.eye(num_classes)[y_val.astype('int')]

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(X_val, y_val))
