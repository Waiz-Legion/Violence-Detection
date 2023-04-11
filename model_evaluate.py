import numpy as np

# Preprocess the test data
violence_test_videos = ['path/to/test/violence/videos']
non_violence_test_videos = ['path/to/test/non-violence/videos']
model = 'path/to/model'
X_test = []
y_test = []
for video_path in violence_test_videos:
    frames = preprocess_data(video_path)
    X_test.append(frames)
    y_test.append(1)
for video_path in non_violence_test_videos:
    frames = preprocess_data(video_path)
    X_test.append(frames)
    y_test.append(0)

X_test = np.array(X_test)
y_test = np.array(y_test)

# Predict the labels of the test dataset
y_pred = model.predict(X_test)
y_pred = np.round(y_pred)

# Calculate the accuracy and other metrics
accuracy = np.mean(y_pred == y_test)
precision = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_pred == 1)
recall = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1)
f1_score = 2 * precision * recall / (precision + recall)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1_score)
