import tensorflow as tf
import PIL.Image
import numpy as np
import argparse
import time

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
    image = PIL.Image.open('2020-12-26 18_04_04.jpg').convert('RGB').resize((width, height),
                                                         PIL.Image.Resampling.LANCZOS)
        # Classify the image
    results = classify_image(interpreter, image)
    print(results   )
    label_id, prob = results[0]
        
    return labels[label_id]
    
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
  
if __name__ == "__main__":
    labels = load_labels("labels.txt")
    interpreter = tf.lite.Interpreter("model.tflite")
    
    print(classify(labels, interpreter))
    
