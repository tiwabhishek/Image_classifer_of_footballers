import joblib  # For loading the trained model
import json  # For handling JSON files
import numpy as np  # For numerical operations
import base64  # For handling base64 encoding/decoding
import cv2  # For image processing
from wavelet import w2d  # Custom wavelet transform function

# Dictionaries to map class names to numbers and vice versa
__class_name_to_number = {}
__class_number_to_name = {}

# Placeholder for the trained model
__model = None

def classify_image(image_base64_data, file_path=None):
    """
    Classifies the image by processing it and passing it through the trained model.

    :param image_base64_data: Base64 encoded image data
    :param file_path: Path to the image file
    :return: List of classification results
    """
    # Get cropped images with at least two eyes detected
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)
    
    result = []
    for img in imgs:
        # Resize the image to 32x32
        scalled_raw_img = cv2.resize(img, (32, 32))
        
        # Apply wavelet transform
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        
        # Combine the raw and wavelet-transformed images
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))
        
        len_image_array = 32*32*3 + 32*32
        final = combined_img.reshape(1, len_image_array).astype(float)
        
        # Predict the class and probability
        result.append({
            'class': class_number_to_name(__model.predict(final)[0]),
            'class_probability': np.around(__model.predict_proba(final)*100, 2).tolist()[0],
            'class_dictionary': __class_name_to_number
        })

    return result

def class_number_to_name(class_num):
    """
    Converts a class number to the corresponding class name.

    :param class_num: Class number
    :return: Class name
    """
    return __class_number_to_name[class_num]

def load_saved_artifacts():
    """
    Loads the saved model and class dictionaries from disk.
    """
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    # Load the class dictionaries from JSON file
    with open("./artifacts/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    global __model
    if __model is None:
        # Load the trained model from a pickle file
        with open('./artifacts/saved_model.pkl', 'rb') as f:
            __model = joblib.load(f)
    print("loading saved artifacts...done")

def get_cv2_image_from_base64_string(b64str):
    """
    Decodes a base64 encoded image string to an OpenCV image.

    :param b64str: Base64 encoded image string
    :return: Decoded OpenCV image
    """
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    """
    Detects faces in the image and returns cropped images of faces with at least two eyes detected.

    :param image_path: Path to the image file
    :param image_base64_data: Base64 encoded image data
    :return: List of cropped face images
    """
    face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    
    return cropped_faces
