import face_recognition
import os
import cv2
import dlib
import dlib.cuda as cuda

#detect if using CUDA
print(cuda.get_num_devices())

print("Dlib CUDA: {}".format(dlib.DLIB_USE_CUDA))

KNOWN_FACES_DIR = "D:/Downloads/ML_Training/face_recognition_training/known_faces"
UNKNOWN_FACES_DIR = "D:/Downloads/ML_Training/face_recognition_training/unknown_faces"
TOLERANCE = 0.6
FRAME_THICKNESS = 2
FONT_THICKNESS = 2
MODEL = "cnn"  #hog

print("loading known faces...")

known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        print(f"Filename: {filename}")
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        encoding = face_recognition.face_encodings(image)[0]
        #print(f"Encoding: {encoding}")
        known_faces.append(encoding)
        known_names.append(name)

print("processing unknown faces...")

for filename in os.listdir(UNKNOWN_FACES_DIR):
    print("Filename is: {}".format(filename))
    image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/{filename}")
    resize_to_w = 1200
    img_w = image.shape[1]
    img_h = image.shape[0]
    image = cv2.resize(image, (resize_to_w, int(resize_to_w*img_h/img_w)))
    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)

    # convert to work with cv2
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)

        # attach matched with known faces list
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print("Match found: {}".format(match))

            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            color = [0, 255, 0]
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            #cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, \
                        0.5, (200, 200, 200), FONT_THICKNESS)
    cv2.imshow(filename, image)
    cv2.waitKey(3000)
    cv2.destroyWindow(filename)
