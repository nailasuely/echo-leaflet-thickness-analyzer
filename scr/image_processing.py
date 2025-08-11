import cv2
import numpy as np
from skimage import color, morphology, filters

def segment_frame(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary_opened = cv2.morphologyEx(binary_closed, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(binary, dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        segmented = cv2.bitwise_and(image, image, mask=mask)
        return segmented, mask, largest_contour
    else:
        return image, np.zeros_like(binary, dtype=np.uint8), None

def extract_skeleton(segmented_image, threshold):
    img_gray = color.rgb2gray(segmented_image)
    img_smooth = filters.gaussian(img_gray, sigma=1)
    binary_image = img_smooth > threshold

    skeleton, _ = morphology.medial_axis(binary_image, return_distance=True)
    return skeleton, binary_image

def prune_skeleton(skeleton, iterations=10):
    pruned = skeleton.copy()
    for _ in range(iterations):
        endpoints = np.zeros_like(pruned, dtype=np.uint8)
        for i in range(1, pruned.shape[0] - 1):
            for j in range(1, pruned.shape[1] - 1):
                if pruned[i, j]:
                    neighborhood = pruned[i-1:i+2, j-1:j+2]
                    # um ponto final tem o pixel central + 1 vizinho
                    if np.sum(neighborhood) == 2:
                        endpoints[i, j] = 1
        pruned[endpoints == 1] = 0
    return pruned