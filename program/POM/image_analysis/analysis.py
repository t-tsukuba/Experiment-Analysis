import cv2
import matplotlib.pyplot as plt


def crack_binary_img(binary_data):
    fig, ax = plt.subplots()

    ax.imshow(binary_data, cmap="gray")
    ax.set_title("crack binary image")
    ax.set_xlabel("$X$")
    ax.set_ylabel("$Y$")
    ax.set_xlim(0, binary_data.shape[1])
    ax.set_ylim(binary_data.shape[0], 0)

    return binary_data


def crack_edge_detect(binary_data):
    edge_detect_img = cv2.Canny(binary_data, threshold1=100, threshold2=255)
    fig, ax = plt.subplots()

    ax.imshow(edge_detect_img, cmap="gray")
    ax.set_title("crack edge detection image")
    ax.set_xlabel("$X$")
    ax.set_ylabel("$Y$")
    ax.set_xlim(0, binary_data.shape[1])
    ax.set_ylim(binary_data.shape[0], 0)

    return edge_detect_img


def edge_contours_count(binary_data):
    edge_detect_img = crack_edge_detect(binary_data)

    contours, _ = cv2.findContours(
        edge_detect_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    count_contours = cv2.drawContours(binary_data, contours, -1, (0, 255, 0), 2)

    return contours, count_contours


def contours_count(binary_data):
    edge_detect_img = cv2.Canny(binary_data, 100, 255)
    contours, _ = cv2.findContours(
        edge_detect_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    count_contours = cv2.drawContours(binary_data, contours, -1, (0, 255, 0), 2)

    return contours, count_contours


def show_contours_count(binary_data):
    edge_detect_img = cv2.Canny(binary_data, 100, 255)
    contours, _ = cv2.findContours(
        edge_detect_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    filtered_contours = [cnt for cnt in contours if len(cnt) > 35]
    filtered_contours_len = len(filtered_contours)

    return filtered_contours_len


def add_contours(binary_data):
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(8, 8))

    img = crack_edge_detect(binary_data)
    filtered_contours_len = show_contours_count(binary_data)

    edge_list = []
    for i in range(filtered_contours_len):
        img = cv2.drawContours(
            cv2.cvtColor(edge_detect_img, cv2.COLOR_GRAY2RGB),
            filtered_contours,
            i,
            (255, 255, 255),
            10,
        )

        edge_list.append(img)

    img = edge_list[0]
    for i in range(len(edge_list) - 1):
        img = cv2.add(img, edge_list[i + 1])

    ax.imshow(img)
    ax.set_title("added edge detect image contour{}".format(i))
    ax.set_xlabel("$X$")
    ax.set_ylabel("$Y$")
    ax.set_xlim(0, binary_data.shape[1])
    ax.set_ylim(binary_data.shape[0], 0)

    plt.tight_layout()
    plt.show()
