import cv2
import face_recognition
import dlib
import urllib.request
import os


def download_predictor():
    # URL для скачивания файла shape_predictor_68_face_landmarks.dat
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

    # Имя файла для сохранения
    file_name = "shape_predictor_68_face_landmarks.dat.bz2"

    # Скачивание файла
    urllib.request.urlretrieve(url, file_name)

    # Распаковка скачанного файла
    os.system(f"bzip2 -d {file_name}")

    # Загрузка файла с помощью dlib и сохранение в формате .dat
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    dlib.save_predictor(predictor, "shape_predictor_68_face_landmarks.dat")


def main(image_path):
    # Проверка наличия файла shape_predictor_68_face_landmarks.dat
    if not os.path.isfile("shape_predictor_68_face_landmarks.dat"):
        print("Downloading shape_predictor_68_face_landmarks.dat...")
        download_predictor()

    # Загрузка изображения
    image = cv2.imread(image_path)

    # Загрузка предварительно обученной модели для выделения ключевых точек лица (dlib)
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Обнаружение лиц с использованием face_recognition
    face_locations = face_recognition.face_locations(image)
    face_landmarks_list = face_recognition.face_landmarks(image, face_locations)

    # Преобразование изображения в формат BGR
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Обход каждого обнаруженного лица и выделение его контура
    for (top, right, bottom, left), landmarks in zip(face_locations, face_landmarks_list):
        # Отрисовка контура лица
        cv2.rectangle(image_bgr, (left, top), (right, bottom), (255, 0, 0), 2)

        # Попробуйте использовать стороннюю библиотеку для определения возраста и пола,
        # такую как https://github.com/yu4u/age-gender-estimation
        # Здесь будет пример вывода возраста и пола
        age, gender = "N/A", "N/A"

        # Вывод информации в консоль
        print("Age:", age)
        print("Gender:", gender)

        # Вывод информации о возрасте и поле на изображении
        cv2.putText(image_bgr, f"Age: {age}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(image_bgr, f"Gender: {gender}", (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Отображение изображения с выделенными лицами и информацией
    cv2.imshow("Detected Faces", image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Пример использования функции
image_path = "your_face.jpg"
main(image_path)
