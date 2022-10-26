import cv2
import numpy as np
import tensorflow as tf

width = 640
height = 480
video = cv2.VideoCapture("C:\\Users\\ACER\\OneDrive\\Máy tính\\Digit Handwritten.webm")
video.set(3,width)
video.set(3,height)

threshold = 0.8
model = tf.keras.models.load_model("./model/mnist_model.h5")
while True:
    ret, frame = video.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    media_blur = cv2.medianBlur(gray,5)
    _, thresh_binary_inv = cv2.threshold(media_blur,127,255,cv2.THRESH_BINARY_INV)
    
    img = cv2.resize(thresh_binary_inv,(28,28))
    img = np.float32(img / 255.0)
    test_x = img.reshape(-1,28,28,1)

    predictions = model.predict(test_x, verbose = False)
    max_prediction_weight = np.amax(predictions[0])
    if max_prediction_weight > threshold:
        test = "{}".format(np.argmax(predictions[0]))
        frame = cv2.putText(frame,test,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    cv2.imshow("Original image",frame)
   


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
video.release()
cv2.destroyAllWindows()
