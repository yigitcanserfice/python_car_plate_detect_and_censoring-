import cv2
import easyocr

# OCR için EasyOCR
reader = easyocr.Reader(['en'])
def detect_and_blur_plate(image_path, output_path):
    # Görüntüyü yükle
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
                plate = image[y:y + h, x:x + w]

                # OCR ile plaka metnini tanı
                result = reader.readtext(plate)

                if result:
                    # Plakayı bulanıklaştır
                    blurred_plate = cv2.GaussianBlur(plate, (15, 15), 30)
                    image[y:y + h, x:x + w] = blurred_plate

                    # Plaka metnini konsola yazdır
                    text = result[0][1]  # İlk bulunan metni alıyoruz
                    print("Plaka:", text)

    # Sonuç görüntüsünü kaydet
    cv2.imwrite(output_path, image)


# Test edelim
detect_and_blur_plate("img/ornek12.jpeg", "output/sansur_ornek12.jpeg")



