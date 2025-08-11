import cv2

def select_roi(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Não foi possível ler o vídeo.")
        return None
    
    ret, frame = cap.read()
    if not ret:
        print("Não foi possível ler o vídeo.")
        cap.release()
        return None
    
    roi = cv2.selectROI("selecione a ROI e pressione ENTER", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    cap.release()

    x, y, w, h = roi
    if w > 0 and h > 0:
        print(f"ROI escolhida: x={x}, y={y}, largura={w}, altura={h}")
        return roi
    else:
        print("nenhuma ROI selecionada.")
        return None