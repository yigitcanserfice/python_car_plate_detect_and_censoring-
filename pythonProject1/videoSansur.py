import cv2
import easyocr

# OCR için EasyOCR
reader = easyocr.Reader(['en'])

def detect_and_blur_plate(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Kenar tespiti
    edges = cv2.Canny(gray, 50, 200)

    # Kontur bulma
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Konturu dikdörtgene çevirme
        approx = cv2.approxPolyDP(contour, 0.03 * cv2.arcLength(contour, True), True)

        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)

            # Plaka boyutlarına uygun olup olmadığını kontrol et
            aspect_ratio = w / float(h)
            if 2 < aspect_ratio < 6:  # Plaka boyut oranı kontrolü
                plate = frame[y:y + h, x:x + w]

                # OCR ile plaka metnini tanı
                result = reader.readtext(plate)

                if result:
                    # Plakayı bulanıklaştır
                    blurred_plate = cv2.GaussianBlur(plate, (15, 15), 30)
                    frame[y:y + h, x:x + w] = blurred_plate


    return frame


def process_video(input_video_path, output_video_path):
    # Video yakalama
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Her kare üzerinde plaka tespiti ve sansürleme işlemi
        processed_frame = detect_and_blur_plate(frame)

        # İşlenen videoyu yazma
        out.write(processed_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


process_video('img/araba_video_6_1.mp4', 'araba_video_sansurlu_6_11.mp4')