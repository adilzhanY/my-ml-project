import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Define the CNN model
def create_model():
    classifier = Sequential()
    classifier.add(Convolution2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Convolution2D(16, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Convolution2D(8, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(128, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(10, activation='softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return classifier

# Load the model
classifier = create_model()
classifier.load_weights('keras_tomato_trained_model_trained.weights.h5')

# Define class labels with corrected descriptions
class_labels = [
    'Tomato has bacterial spot',
    'Tomato has early blight',
    'Tomato is healthy',
    'Tomato has late blight',
    'Tomato has leaf mold',
    'Tomato has septoria leaf spot',
    'Tomato has spider mites (Two-spotted spider mite)',
    'Tomato has target spot',
    'Tomato has mosaic virus',
    'Tomato has yellow leaf curl virus'
]

st.title("Tomato Leaf Disease Classification")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    img = image.resize((128, 128))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    predictions = classifier.predict(img)
    st.write("Prediction Probabilities:", predictions)

    # Get the class with the highest probability
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_label = class_labels[predicted_class_index]
    st.write("Predicted Class:", predicted_class_label)
