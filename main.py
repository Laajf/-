import cv2
import numpy as np
import requests
from pathlib import Path
import time
import webbrowser
import pyautogui
import pyglet
#from scan import main

yolo_weights_url = "https://pjreddie.com/media/files/yolov3.weights"
yolo_cfg_url = "https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true"
coco_names_url = "https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true"
haarcascade_url = "https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml?raw=true"

# Пути для сохранения скачанных файлов
yolo_weights_path = "yolov3.weights"
yolo_cfg_path = "yolov3.cfg"
coco_names_path = "coco.names"
haarcascade_path = "haarcascade_frontalface_default.xml"


# Проверка на размыленность
def is_image_blurry(image_path, threshold=100):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()

    return fm < threshold


# Функция для скачивания файла
def download_file(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as file:
        file.write(response.content)


# Проверяем наличие файлов, если их нет, скачиваем
if not Path(yolo_weights_path).exists():
    print(f"Скачиваем {yolo_weights_path}")
    download_file(yolo_weights_url, yolo_weights_path)

if not Path(yolo_cfg_path).exists():
    print(f"Скачиваем {yolo_cfg_path}")
    download_file(yolo_cfg_url, yolo_cfg_path)

if not Path(coco_names_path).exists():
    print(f"Скачиваем {coco_names_path}")
    download_file(coco_names_url, coco_names_path)

if not Path(haarcascade_path).exists():
    print(f"Скачиваем {haarcascade_path}")
    download_file(haarcascade_url, haarcascade_path)

# Загружаем предварительно обученные модели
net = cv2.dnn.readNet(yolo_weights_path, yolo_cfg_path)
face_cascade = cv2.CascadeClassifier(haarcascade_path)

# Загружаем классы объектов
with open(coco_names_path, "r") as f:
    classes = f.read().strip().split("\n")

# Получаем видео с камеры
video = cv2.VideoCapture(0)

# Изображение вашего лица
known_face = cv2.imread("your_face.jpg", cv2.IMREAD_GRAYSCALE)

# Значение порога схожести изображений
threshold = 0.55

while cv2.waitKey(1) < 0:
    # Получаем очередной кадр с камеры
    has_frame, frame = video.read()

    # Если кадра нет
    if not has_frame:
        # Останавливаемся и выходим из цикла
        cv2.waitKey()
        break

    # Распознаем лица на кадре
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Если обнаружено лицо
    if len(faces) > 0:
        # Перебираем все обнаруженные лица
        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]

            # Сравниваем с известным лицом
            result = cv2.matchTemplate(face_roi, known_face, cv2.TM_CCOEFF_NORMED)
            _, confidence, _, _ = cv2.minMaxLoc(result)

            if confidence > threshold:
                print("Неизвестная обаба")

                filename = f"intruder45.jpg"
                cv2.imwrite(filename, frame)

                song = pyglet.media.load('file.mp3')
                song.play()

                #main(filename)

                time.sleep(2)
            else:
                print("Дудник Артем Дмитриевич")

    # Получаем высоту и ширину кадра
    height, width = frame.shape[:2]

    # Подготавливаем кадр для обнаружения объектов
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Получаем имена слоев модели YOLO
    layer_names = net.getUnconnectedOutLayersNames()

    # Обнаруживаем объекты на кадре
    outs = net.forward(layer_names)

    # Обработка результатов обнаружения
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, classes[class_id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Выводим картинку с камеры с обозначенными объектами
    cv2.imshow("Object detection", frame)
    time.sleep(1)
