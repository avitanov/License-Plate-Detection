import cv2
import numpy as np
import pytesseract
import os
from object_detection import process_images

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def read_plate_from_images(input_folder, output_folder):
    if not os.path.isdir(input_folder):
        print(f"Фолдерот {input_folder} не постои.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.startswith("test") and f.endswith(".jpg")]
    if not image_files:
        print(f"Нема слики во {input_folder} со формат test*.jpg")
        return

    for image_file in sorted(image_files):
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Не може да се вчита слика од: {image_path}")
            continue

        # Претвaрање во сива скала
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Зголемување на димензиите (зумирање)
        zoomed = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # Отстранување на шум со median blur
        median = cv2.medianBlur(zoomed, 3)

        # Otsu Threshold за подобрување на контрастот
        _, otsu = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Морфолошки операции за чистење на сликата
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        processed = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

        # Острење на сликата
        kernel_sharpen = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
        sharpened = cv2.filter2D(processed, -1, kernel_sharpen)

        # Tesseract OCR за читање на текстот
        config = r"--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        text = pytesseract.image_to_string(sharpened, lang='eng', config=config).strip()

        print(f"[{image_file}] Прочитан текст: {text}")

        # Додавање на прочитаниот текст на оригиналната слика
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (50, 30)
        font_scale = 1.3
        font_color = (255, 0, 0)
        thickness = 2

        cv2.putText(image, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)


        output_path = os.path.join(output_folder, f"annotated_{image_file}")
        success = cv2.imwrite(output_path, image)

        if success:
            print(f"Сликата успешно зачувана во: {output_path}")
        else:
            print(f"Грешка при зачувување на сликата: {output_path}")

        # Прикажување на финалната слика со прочитаниот текст
        cv2.imshow("Detected Plate", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    process_images(image_dir="../data")
    read_plate_from_images(input_folder="process_images", output_folder="output_images")
