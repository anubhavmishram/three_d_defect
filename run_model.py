import cv2
import numpy as np
import pickle

# Load the saved model
with open('my_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load and preprocess the test image
img = cv2.imread('test_image.jpg')
img = cv2.resize(img, (224, 224))
img = img.astype('float32') / 255.0

# Run inference and get the output probabilities
probs = model.predict_proba(np.expand_dims(img, axis=0))

# Print the predicted class
class_idx = np.argmax(probs)
print('Predicted class:', class_idx)
