import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

priyanshu_image = face_recognition.load_image_file(r"C:\Users\syed aafreen\OneDrive\Desktop\pics\face\priyanshu.jpg")
priyanshu_encoding = face_recognition.face_encodings(priyanshu_image)[0]

prachi_image = face_recognition.load_image_file(r"C:\Users\syed aafreen\OneDrive\Desktop\pics\face\prachi.jpg")
prachi_encoding = face_recognition.face_encodings(prachi_image)[0]

Manorama_image = face_recognition.load_image_file(r"C:\Users\syed aafreen\OneDrive\Desktop\pics\face\Dr. Manorama Mam.jpg")
Manorama_encoding= face_recognition.face_encodings(Manorama_image)[0]

prashant_image = face_recognition.load_image_file(r"C:\Users\syed aafreen\OneDrive\Desktop\pics\face\prashant.jpg")
prashant_encoding = face_recognition.face_encodings(prashant_image)[0]

sneha_image = face_recognition.load_image_file(r"C:\Users\syed aafreen\OneDrive\Desktop\pics\face\sneha.jpg")
sneha_encoding = face_recognition.face_encodings(sneha_image)[0]

ratan_image = face_recognition.load_image_file(r"C:\Users\syed aafreen\OneDrive\Desktop\pics\face\tata.jpeg")
ratan_encoding = face_recognition.face_encodings(ratan_image)[0]

billgates_image = face_recognition.load_image_file(r"C:\Users\syed aafreen\OneDrive\Desktop\pics\face\billgates.jpg")
billgates_encoding = face_recognition.face_encodings(billgates_image)[0]

sundar_image = face_recognition.load_image_file(r"C:\Users\syed aafreen\OneDrive\Desktop\pics\face\sundarpichai.jpg")
sundar_encoding = face_recognition.face_encodings(sundar_image)[0]

salman_image = face_recognition.load_image_file(r"C:\Users\syed aafreen\OneDrive\Desktop\pics\face\salman.jpg")
salman_encoding = face_recognition.face_encodings(salman_image)[0]

samay_image = face_recognition.load_image_file(r"C:\Users\syed aafreen\OneDrive\Desktop\pics\face\samay.jpg")
samay_encoding = face_recognition.face_encodings(samay_image)[0]

shahrukh_image = face_recognition.load_image_file(r"C:\Users\syed aafreen\OneDrive\Desktop\pics\face\shahrukh.jpg")
shahrukh_encoding = face_recognition.face_encodings(shahrukh_image)[0]

rani_image = face_recognition.load_image_file(r"C:\Users\syed aafreen\OneDrive\Desktop\pics\face\ranimukherjee.jpg")
rani_encoding = face_recognition.face_encodings(rani_image)[0]

aishwarya_image = face_recognition.load_image_file(r"C:\Users\syed aafreen\OneDrive\Desktop\pics\face\aishwarya.jpg")
aishwarya_encoding = face_recognition.face_encodings(aishwarya_image)[0]

preity_image = face_recognition.load_image_file(r"C:\Users\syed aafreen\OneDrive\Desktop\pics\face\preityzinta.jpg")
preity_encoding = face_recognition.face_encodings(preity_image)[0]

sameer_image = face_recognition.load_image_file(r"C:\Users\syed aafreen\OneDrive\Desktop\pics\face\sameer.jpg")
sameer_encoding = face_recognition.face_encodings(sameer_image)[0]



known_face_encodings = [
    priyanshu_encoding,
    prachi_encoding ,
    Manorama_encoding,
     prashant_encoding,
     sneha_encoding,
     ratan_encoding,
     billgates_encoding,
     sundar_encoding,
     salman_encoding,
     samay_encoding,
     shahrukh_encoding,
     rani_encoding,
     aishwarya_encoding,
     preity_encoding,
     sameer_encoding]
known_face_names = [
    "priyanshu",
    "prachi",
    "Dr Manorama Mam",
    "prashant",
    "sneha",
    "ratan",
    "billgates",
    "sundarpichai",
    "salmankhan",
    "samay",
    "shahrukh",
    "ranimukherjee",
    "aishwaryarai",
    "preityzinta",
    "sameer"]
students = known_face_names.copy()
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not open video file.")
    exit()
import os
from datetime import datetime
# Get the current date
current_date = datetime.now().strftime('%Y-%m-%d')
# Define the directory and file name
directory = 'data'
file_name = f"{current_date}.csv"
file_path = os.path.join(directory, file_name)
# Create the directory if it doesn't exist
os.makedirs(directory, exist_ok=True)
try:
    # Open the file in write mode
    with open(file_path, 'w+', newline='') as f:
        # Do something with the file
        pass
    print(f"File '{file_name}' created successfully.")
except FileNotFoundError as e:
    print("Error:", e)
while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            if name in students:
                students.remove(name)
                current_time = datetime.now().strftime("%H-%M-%S")
                with open('attendance.csv', 'a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([name, current_time])
        face_names.append(name)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)
    cv2.imshow("attendance system", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()                  