import tensorflow as tf
import PIL.Image
import numpy as np
import argparse
import time
import cv2

# Function to set the input tensor for the interpreter
def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

# Function to classify the image
def classify_image(interpreter, image, top_k=1):
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)
    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]

# Function to load labels from file
def load_labels(path):
    with open(path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

# Function to classify and annotate the image
def classify(labels, interpreter, image):
    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    image_resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
    cv2.imshow('resize', image_resized)
    results = classify_image(interpreter, image_resized)
    label_id, prob = results[0]
    label_text = f'{labels[label_id]} ({prob*100:.2f}%)'
    print( results[0])
    print(labels[label_id])
    annotated_image = image.copy()
    cv2.putText(annotated_image, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return annotated_image

# Function to draw bounding box and handle mouse events
def capture_frame(event, x, y, flags, param):
    global captured_frame, start_point, end_point, drawing, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        captured_frame = frame[start_point[1]:end_point[1], start_point[0]:end_point[0]]

# Initialize the video capture
cap = cv2.VideoCapture(3)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

cv2.namedWindow('USB Camera')
cv2.setMouseCallback('USB Camera', capture_frame)

captured_frame = None
drawing = False
start_point = (0, 0)
end_point = (0, 0)

if __name__ == "__main__":
    labels = load_labels("labels.txt")
    interpreter = tf.lite.Interpreter(model_path="model.tflite")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        if drawing:
            cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)

        cv2.imshow('USB Camera', frame)

        if captured_frame is not None:
            result_frame = classify(labels, interpreter, captured_frame)
            cv2.imshow('Detection Result', result_frame)
            captured_frame = None

        key = cv2.waitKey(1)
        if key == 27:  # Press 'ESC' to exit
            break

    cap.release()
    cv2.destroyAllWindows()
