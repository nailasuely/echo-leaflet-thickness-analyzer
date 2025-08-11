import cv2
import numpy as np

def identify_flexion_point(contour):
    if contour is None or len(contour) < 3:
        return None
    
    moments = cv2.moments(contour)
    if moments['m00'] == 0:
        return None
        
    curvature_points = []
    for i in range(len(contour)):
        p1 = contour[i-1][0]
        p2 = contour[i][0]
        p3 = contour[(i + 1) % len(contour)][0]

        angle = np.abs(np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - 
                       np.arctan2(p1[1] - p2[1], p1[0] - p2[0]))
        angle = min(angle, 2 * np.pi - angle)
        curvature_points.append((angle, p2))

    if not curvature_points:
        return None
        
    max_curvature_point_info = max(curvature_points, key=lambda x: x[0])
    best_point = max_curvature_point_info[1]

    return tuple(best_point)

def divide_segments(flexion_point, contour, width=30):
    x_flex, _ = flexion_point
    
    x_start = min(contour[:, 0, 0])
    tip = (x_start, x_flex)

    mid_start = x_flex - width
    mid_end = x_flex + width
    mid = (mid_start, mid_end)
    
    base_start = x_flex + width
    base_end = x_flex + 2 * width
    base = (base_start, base_end)

    return base, mid, tip

def calculate_segment_thickness(skeleton, contour, segment):
    x_start, x_end = segment
    thicknesses = []

    height, width = skeleton.shape
    x_start = max(int(x_start), 0)
    x_end = min(int(x_end), width)

    for y in range(height):
        for x in range(x_start, x_end):
            if skeleton[y, x]:
                distance_to_contour = cv2.pointPolygonTest(contour, (float(x), float(y)), True)
                if distance_to_contour > 0:
                    thickness = distance_to_contour * 2
                    thicknesses.append(thickness)
    
    mean_thickness = np.mean(thicknesses) if thicknesses else 0
    max_thickness = np.max(thicknesses) if thicknesses else 0
    std_dev = np.std(thicknesses) if thicknesses else 0

    return mean_thickness, max_thickness, std_dev