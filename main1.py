import cv2 as cv
import numpy as np

people = ['Elon Musk', 'Jeff Bezos', 'Kamala Harris', 'Leonardo DiCaprio', 'Narendra Modi']
#features = np.load('features.npy', allow_pickle=True)
#labels = np.load('labels.npy')

haar = cv.CascadeClassifier('C:/Users/msi/Python/opencv/faceDetect.xml')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print('Cannot open camera!')
    exit()
while True:
    ret, frame = cap.read()
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faceD = cv.CascadeClassifier('C:/Users/msi/Python/opencv/faceDetect.xml')
    detect = faceD.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)

    for (x,y,w,h) in detect:
        roi = gray[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(roi)
        cv.putText(frame, str(people[label]), (20,30), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), 2)
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), thickness=1)

    cv.imshow("Detected", frame)
    
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

cap.release()
cv.destroyAllWindows()


