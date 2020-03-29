import face_recognition
import os, time
import cv2
import imutils
from imutils.video import VideoStream

KNOWN_FACES_DIR = "D:/Downloads/ML_Training/face_recognition_training/known_faces"
#UNKNOWN_FACES_DIR = "D:/Downloads/ML_Training/face_recognition_training/unknown_faces"
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
video = cv2.VideoCapture(0)  #could pu in a filename "D:/Videos/Captures/YDXJ0675.mp4"
# initialize the video stream and allow the cammera sensor to warmup
print("Starting video stream...")
#video = VideoStream(src=1).start()
time.sleep(2.0)

while True:
    #print("Filename is: {}".format(filename))
    #image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/{filename}")

    ret, image = video.read()
    image = imutils.resize(image, width=400)
    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)

    # convert to work with cv2 for images, no need for video feed
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
        cv2.startWindowThread()
        cv2.namedWindow("vid feed")
    cv2.imshow("vid feed", image)
    key = cv2.waitKey(1) & 0xFF
    #Break if "q" pressed
    if  key == ord("q"):
        break
        #cv2.waitKey(3000)
cv2.destroyAllWindows()
video.stop()

