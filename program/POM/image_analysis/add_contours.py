import cv2


def get_added_contours_img(filtered_contours, edge_img, line_width):
    base_img = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2RGB)
    # それぞれの輪郭を描画した画像のリスト
    contour_imgs = []
    for i in range(len((filtered_contours))):
        contour_img = cv2.drawContours(
            base_img,
            filtered_contours,  # 輪郭を保存したリスト
            i,  # リストの何番目か
            (255, 255, 255),  # 白
            line_width,  # 線の太さ
        )

        contour_imgs.append(contour_img)
    # リストの最初の画像に他の画像を足し合わせる
    added_img = contour_imgs[0]
    for contour_img in contour_imgs[1:]:
        added_img = cv2.add(added_img, contour_img)

    return added_img


def get_highlight_edge_img(binary_img, min_contour_size, line_width):
    # thresholdを変えても変化がなかった
    edge_img = cv2.Canny(binary_img, threshold1=100, threshold2=255)
    contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if len(cnt) > min_contour_size]
    added_img = get_added_contours_img(filtered_contours, edge_img, line_width)

    return added_img
