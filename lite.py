import cv2  # Install opencv-python
import numpy as np
import tensorflow as tf  # Required for TensorFlow Lite Interpreter

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on the default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcam's image
    ret, image = camera.read()

    # Resize the raw image into (224-height, 224-width) pixels
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image_resized)

    # Preprocess the image: Convert it to a numpy array, normalize, and reshape
    input_data = np.asarray(image_resized, dtype=np.float32)
    input_data = (input_data / 127.5) - 1  # Normalize to [-1, 1]
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    index = np.argmax(output_data)
    class_name = class_names[index]
    confidence_score = output_data[0][index]

    # Print prediction and confidence score
    print("Class:", class_name.strip(), end=" ")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII code for the 'esc' key
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
