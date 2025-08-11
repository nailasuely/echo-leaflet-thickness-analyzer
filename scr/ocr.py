import cv2
import pytesseract

try:
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except Exception as e:
    print("aviso: O caminho para o Tesseract OCR não foi encontrado ou está incorreto.")

def extract_text_from_roi(frame, roi_percentage=(0.1, 0.3)):
    height, width = frame.shape[:2]
    roi_height = int(height * roi_percentage[0])
    roi_width = int(width * roi_percentage[1])
    
    roi = frame[0:roi_height, 0:roi_width]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    text = pytesseract.image_to_string(binary, config="--psm 6")
    return text.strip()