import cv2


def img_to_grayscasle(input_filepath):
    raw_img = cv2.imread(input_filepath)
    gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

    return gray_img


def grayscale_to_binary(gray, th):
    # 閾値を設定して二値化
    _, binary_img = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)

    return binary_img


def img_to_binary(input_filepath, threshold):
    gray_img = img_to_grayscasle(input_filepath)
    binary_img = grayscale_to_binary(gray_img, threshold)

    return binary_img
