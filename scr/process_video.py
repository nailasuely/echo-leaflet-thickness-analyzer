import cv2
import numpy as np
import os
from roi_selector import select_roi
from image_processing import segment_frame, extract_skeleton, prune_skeleton
from analysis import identify_flexion_point, divide_segments, calculate_segment_thickness
from ocr import extract_text_from_roi

def process_video(video_path, valve_roi, threshold, scale_px_per_cm):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    results = {
        'base_means': [], 'mid_means': [], 'tip_means': [],
        'base_maxes': [], 'mid_maxes': [], 'tip_maxes': []
    }
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1     
        if frame_count == 1:
            extracted_text = extract_text_from_roi(frame)
            print(f"Texto extraído do primeiro frame: {extracted_text}")
        x, y, w, h = valve_roi
        leaflet_roi = frame[y:y+h, x:x+w]

        segmented, _, contour = segment_frame(leaflet_roi)
        if contour is None:
            continue

        skeleton, _ = extract_skeleton(segmented, threshold)
        pruned_skeleton = prune_skeleton(skeleton)
        flexion_point = identify_flexion_point(contour)

        if flexion_point is None:
            continue

        base_seg, mid_seg, tip_seg = divide_segments(flexion_point, contour)
        
        base_mean, base_max, _ = calculate_segment_thickness(pruned_skeleton, contour, base_seg)
        mid_mean, mid_max, _ = calculate_segment_thickness(pruned_skeleton, contour, mid_seg)
        tip_mean, tip_max, _ = calculate_segment_thickness(pruned_skeleton, contour, tip_seg)

        if all([base_mean > 0, mid_mean > 0, tip_mean > 0]):
            results['base_means'].append(base_mean)
            results['mid_means'].append(mid_mean)
            results['tip_means'].append(tip_mean)
            results['base_maxes'].append(base_max)
            results['mid_maxes'].append(mid_max)
            results['tip_maxes'].append(tip_max)

    cap.release()
    cv2.destroyAllWindows()
    print_summary_report(results, scale_px_per_cm)

def print_summary_report(results, scale):
    pixel_to_mm = 10 / scale

    avg_mean_base = np.mean(results['base_means']) if results['base_means'] else 0
    avg_mean_mid = np.mean(results['mid_means']) if results['mid_means'] else 0
    avg_mean_tip = np.mean(results['tip_means']) if results['tip_means'] else 0
    
    avg_max_base = np.mean(results['base_maxes']) if results['base_maxes'] else 0
    avg_max_mid = np.mean(results['mid_maxes']) if results['mid_maxes'] else 0
    avg_max_tip = np.mean(results['tip_maxes']) if results['tip_maxes'] else 0

    std_mean_base = np.std(results['base_means']) if results['base_means'] else 0
    std_mean_mid = np.std(results['mid_means']) if results['mid_means'] else 0
    std_mean_tip = np.std(results['tip_means']) if results['tip_means'] else 0

    print("\n--------------------------------------------------\n")
    print("resultados Finais (apenas frames válidos)")
    print(f"Média de espessura - Base: {avg_mean_base:.2f} px ({avg_mean_base * pixel_to_mm:.2f} mm) | Máxima Média: {avg_max_base:.2f} px ({avg_max_base * pixel_to_mm:.2f} mm)")
    print(f"Média de espessura - Segmento Médio: {avg_mean_mid:.2f} px ({avg_mean_mid * pixel_to_mm:.2f} mm) | Máxima Média: {avg_max_mid:.2f} px ({avg_max_mid * pixel_to_mm:.2f} mm)")
    print(f"Média de espessura - Extremidade: {avg_mean_tip:.2f} px ({avg_mean_tip * pixel_to_mm:.2f} mm) | Máxima Média: {avg_max_tip:.2f} px ({avg_max_tip * pixel_to_mm:.2f} mm)")

    ratio = lambda a, b: a / b if b != 0 else 0
    print("\nRelações de Espessura Média:")
    print(f"Base/Médio: {ratio(avg_mean_base, avg_mean_mid):.2f}")
    print(f"Ponta/Base: {ratio(avg_mean_tip, avg_mean_base):.2f}")

    print("\nRelações de Espessura Máxima Média:")
    print(f"Base/Médio: {ratio(avg_max_base, avg_max_mid):.2f}")
    print(f"Ponta/Base: {ratio(avg_max_tip, avg_max_base):.2f}")
    
    print("\n--------------------------------------------------\n")
    print("Desvio Padrão entre frames:")
    print(f"Desvio Padrão (px) - Base: {std_mean_base:.2f} | Médio: {std_mean_mid:.2f} | Ponta: {std_mean_tip:.2f}")
    print(f"Desvio Padrão (mm) - Base: {std_mean_base * pixel_to_mm:.2f} | Médio: {std_mean_mid * pixel_to_mm:.2f} | Ponta: {std_mean_tip * pixel_to_mm:.2f}")


if __name__ == "__main__":
    video_filepath = "video.mp4" 
    segmentation_threshold = 0.5
    scale_px_per_cm = 52.0
    
    if not os.path.exists(video_filepath):
        print(f"Arquivo de vídeo não encontrado em: {video_filepath}")
    else:
        selected_roi = select_roi(video_filepath)
        if selected_roi:
            process_video(video_filepath, selected_roi, segmentation_threshold, scale_px_per_cm)