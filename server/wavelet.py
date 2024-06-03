import numpy as np  # For numerical operations
import pywt  # For wavelet transformations
import cv2  # For image processing

def w2d(img, mode='haar', level=1):
    """
    Apply wavelet transform to the input image and process the coefficients.

    :param img: Input image
    :param mode: Wavelet mode (default is 'haar')
    :param level: Level of decomposition (default is 1)
    :return: Processed image after wavelet transformation
    """
    imArray = img
    
    # Convert to grayscale
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    
    # Convert to float
    imArray = np.float32(imArray)
    imArray /= 255  # Normalize pixel values to the range [0, 1]
    
    # Compute wavelet coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    
    # Process coefficients (set approximation coefficients to zero)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0  # Set approximation coefficients to zero
    
    # Reconstruction from modified coefficients
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255  # Scale pixel values back to the range [0, 255]
    imArray_H = np.uint8(imArray_H)  # Convert to unsigned 8-bit integer

    return imArray_H
