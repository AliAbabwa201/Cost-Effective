import tensorflow as tf
import PIL.Image
import numpy as np
import argparse
import time
import numpy as np
import cv2

def classify(labels, interpreter):
    """
    Takes the image of the object and returns its name ('plastic', 'metal', 'carton') and the probability of it. 
    If there is no object, returns 'None'.
    """    
    # Setting up the model and the labels
    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    
    # Opening camera and capturing the image
        # This detected.jpg will always override in every recycle, so do not worry about memory issues.
    #image = PIL.Image.open('metal18.jpg').convert('RGB').resize((width, height),PIL.Image.Resampling.LANCZOS)
    image = cv2.resize(captured_frame,(width, height),PIL.Image.Resampling.LANCZOS)
        # Classify the image
    results = classify_image(interpreter, image)
    print(results   )
    label_id, prob = results[0]

    label_text = f'{labels[label_id]} ({np.max(prob)*100:.2f}%)'
    annotated_image = image.copy()
    cv2.putText(annotated_image, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    print(labels[label_id])
    return annotated_image
        
    #return labels[label_id]

# Function to predict and annotate the image
def predict_and_annotate(image):
    preprocessed_image = preprocess_image(image, (70, 70))  # Adjust target size to (70, 70)
    
    yhat = model.predict(preprocessed_image)
    print(yhat)
    predicted_class_index = np.argmax(yhat, axis=1)
    label_text = f'{class_names[predicted_class_index][0]} ({np.max(yhat)*100:.2f}%)'
    annotated_image = image.copy()
    cv2.putText(annotated_image, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return annotated_image
    
def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))
  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]
  
def load_labels(path):
    """Loads the file path of the labels file."""
    with open(path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}
  

# Initialize the video capture
cap = cv2.VideoCapture(3)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

cv2.namedWindow('USB Camera')

# Variable to store the frame when clicked
captured_frame = None

# Mouse callback function to capture frame on click

def capture_frame(event, x, y, flags, param):
    global captured_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        captured_frame = frame.copy()


global frame
cv2.setMouseCallback('USB Camera', capture_frame)

if __name__ == "__main__":
    labels = load_labels("labels.txt")
    interpreter = tf.lite.Interpreter("model.tflite")
    
    #print(classify(labels, interpreter))
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow('USB Camera', frame)

        if captured_frame is not None:
            cv2.imshow('Copy', captured_frame)
            result_frame = classify(labels, interpreter)
            cv2.imshow('Detection Result', result_frame)
            captured_frame = None  # Reset captured_frame after processing

        key = cv2.waitKey(1)
        if key == 27:  # Press 'ESC' to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    
