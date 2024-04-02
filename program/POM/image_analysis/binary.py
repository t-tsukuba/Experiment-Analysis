import cv2


def grayscale_to_binary(input_filepath, threshold_value=90):
    src = cv2.imread(input_filepath)
    _, gray_th = cv2.threshold(src, threshold_value, 255, cv2.THRESH_BINARY)

    return gray_th
