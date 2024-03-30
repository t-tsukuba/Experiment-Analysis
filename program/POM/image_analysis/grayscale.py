import cv2


def img_to_grayscasle(input_filepath):
    src = cv2.imread(input_filepath)
    cvtcolor_result = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    return cvtcolor_result
