import cv2 as cv
import sys
import numpy as np
import os

# Иницијализација на параметри
confThreshold = 0.5  # Праг за доверба (колку моделот е сигурен во предвидувањето)
nmsThreshold = 0.4  # Праг за потиснување на преклопување на кутии (Non-Max Suppression)
inpWidth = 416  # Ширина на влезната слика за YOLO моделот
inpHeight = 416  # Висина на влезната слика за YOLO моделот

# Дефинирање на класите директно во кодот
classes = ["License Plate"]  # Единствената класа е „Регистарска Табличка“
license_plate_class_id = 0  # ID за класата „Регистарска Табличка“ (секогаш е 0 бидејќи е единствена класа)

# Вчитување на YOLO моделот (конфигурација и тежини)
modelConfiguration = "../model/config/darknet-yolov3.cfg"  # Патека до YOLO конфигурациската датотека
modelWeights = "../model/weights/model.weights"  # Патека до YOLO тежините
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)  # Вчитување на YOLO моделот
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)  # Користење на OpenCV како бекенд
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)  # Користење на CPU за обработка

# Функција за добивање на имињата на излезните слоеви
def getOutputsNames(net):
    layersNames = net.getLayerNames()  # Вчитување на сите имиња на слоевите
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Функција за обработка на резултатите, цртање на кутии и зачувување на исечените слики
def postprocess(frame, outs, save_path, counter):
    frameHeight = frame.shape[0]  # Висина на сликата
    frameWidth = frame.shape[1]  # Ширина на сликата
    classIds = []  # Листа за ID-ја на класи
    confidences = []  # Листа за доверба на детекцијата
    boxes = []  # Листа за координати на кутии

    for out in outs:  # За секој излезен слој
        for detection in out:  # За секоја детекција
            scores = detection[5:]  # Се земаат резултатите за класи
            classId = np.argmax(scores)  # ID на класата со највисок резултат
            confidence = scores[classId]  # Доверба на таа класа
            if confidence > confThreshold and classId == license_plate_class_id:  # Ако довербата е поголема од прагот и класата е „License Plate“
                center_x = int(detection[0] * frameWidth)  # Центар X координата на кутијата
                center_y = int(detection[1] * frameHeight)  # Центар Y координата на кутијата
                width = int(detection[2] * frameWidth)  # Ширина на кутијата
                height = int(detection[3] * frameHeight)  # Висина на кутијата
                left = int(center_x - width / 2)  # Лева координата
                top = int(center_y - height / 2)  # Горна координата
                classIds.append(classId)  # Додавање на ID-то на класата
                confidences.append(float(confidence))  # Додавање на довербата
                boxes.append([left, top, width, height])  # Додавање на кутијата

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)  # Примена на Non-Max Suppression
    for i in indices:  # За секоја селектирана кутија
        i = i[0]  # Индекс на кутијата
        box = boxes[i]  # Координати на кутијата
        left, top, width, height = box  # координатите
        cropped = frame[top:top + height, left:left + width]  # Исечена слика од кутијата
        cv.imwrite(f"{save_path}/test{counter}.jpg", cropped)  # Зачувување на исечената слика
        counter += 1
    return counter

# Главна функција за обработка на слики
def process_images(image=None, image_dir=None, output_dir="process_images"):
    counter = 1
    os.makedirs(output_dir, exist_ok=True)

    if image:  # Ако е дадена поединечна слика
        if not os.path.isfile(image):
            print(f"Input image file {image} doesn't exist")
            sys.exit(1)
        frame = cv.imread(image)
        blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)  # Претворање на сликата во blob за YOLO
        net.setInput(blob)  # Подесување на blob како влез за мрежата
        outs = net.forward(getOutputsNames(net))  # излези од YOLO
        counter = postprocess(frame, outs, output_dir, counter)  # Обработка на излезите
        print(f"Processed image saved in {output_dir}")

    elif image_dir:  # Ако е даден директориум со слики
        if not os.path.isdir(image_dir):
            print(f"Input image directory {image_dir} doesn't exist")
            sys.exit(1)
        for image_file in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_file)
            if os.path.isfile(image_path):
                frame = cv.imread(image_path)
                blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)  # Претворање на blob
                net.setInput(blob)  # Подесување на blob како влез за мрежата
                outs = net.forward(getOutputsNames(net))  # излезите од YOLO
                counter = postprocess(frame, outs, output_dir, counter)  # Обработка на излезите
        print(f"Processed images saved in {output_dir}")
    else:
        print("Either an image or a directory must be provided.")

