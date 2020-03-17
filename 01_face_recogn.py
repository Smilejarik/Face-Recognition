import face_recognition
import os
import cv2

KNOWN_FACES_DIR = "D:\Downloads\ML_Training\\face_recognition_training\known_faces"
UNKNOWN_FACES_DIR = "D:\Downloads\ML_Training\\face_recognition_training\unknown_faces"
TOLERANCE = 0.6
FRAME_THICKNESS = 2
FONT_THICKNESS = 2
MODEL = "cnn"  #hog

print("loading known faces...")

known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)

print("processing unknown faces...")

for filename in os.listdir(UNKNOWN_FACES_DIR):
    print("Filename is: {}".format(filename))
    image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/{filename}")
    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)

    # convert to work with cv2
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        
