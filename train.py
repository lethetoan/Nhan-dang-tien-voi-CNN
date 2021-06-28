from os import listdir
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

raw_folder = "C:/Users/ACER/MiAI_Money_Classify/data/"
def save_data(raw_folder=raw_folder):

    dest_size = (128, 128)
    print("Bắt đầu xử lý ảnh...")

    pixels = []
    labels = []

    # chạy các vòng lặp để lấy được ảnh nhé
    for folder in listdir(raw_folder):   # Truy cập vào folder chứa các file ảnh
        if folder!='.DS_Store':          # Dùng trên Mac nhé
            print("Folder=",folder)
            # Lặp qua các file trong từng thư mục chứa các ảnh
            for file in listdir(raw_folder  + folder):
                if file!='.DS_Store':
                    print("File=", file)
                    pixels.append( cv2.resize(cv2.imread(raw_folder  + folder +"/" + file),dsize=(128,128)))
                    labels.append( folder)

    pixels = np.array(pixels)
    labels = np.array(labels)#.reshape(-1,1)

    from sklearn.preprocessing import LabelBinarizer
    encoder = LabelBinarizer()
    labels = encoder.fit_transform(labels)             # Chuyển về dạng one - hot - encoder 
    print(labels)

    file = open('pix.data', 'wb')             # Lưu pixels và labels vào file pix.data
    # dump information to that file
    pickle.dump((pixels,labels), file)
    # close the file
    file.close()
    return
save_data()                                   # Nhớ gọi hàm không là không hiện được file pix.data đâu nhé

def load_data():                      # Hàm load data để mở file pix.data
    file = open('pix.data', 'rb')

    # dump information to that file
    (pixels, labels) = pickle.load(file)

    # close the file
    file.close()

    print(pixels.shape)
    print(labels.shape)


    return pixels, labels


X,y = load_data()                           # gọi hàm load_data để tạo biến X và y ( X là ảnh, y là nhãn)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=100)  # Chia hàm train và hàm test để train và test độ chính xác của model

print(X_train.shape)
print(y_train.shape)
X_train = X_train.astype('float')*1./255           # vì mình muốn dùng hàm softmax để tìm xác suất vào class mình mong muốn, nếu chưa hiểu có thể nhắn mình nhé
X_test = X_test.astype('float')*1./255

# SET MODEL

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape= (128, 128, 3)))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.5))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.5))
model.add(Flatten())
model.add(Dense(4, activation='softmax'))         
model.summary()          # Xem Output và Param cho vui :)))

# TRAIN MODEL
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # Nhớ chạy lệnh compile, vì đây là lệnh chứa hàm mất mát
model.fit(X_train, y_train, batch_size=64, epochs = 15,validation_data=(X_test, y_test)) # Hàm này dùng để train thôi, lấy epochs và batch-size theo thưc nghiệm để lấy được kết quả tốt nhất nhé.

# Mình recommend dùng Colab nhé, chứ máy không khỏe là sẽ đơ luôn. ( Cái này mình chạy trên máy luôn, hơi lâu)
                                          # OK đây chỉ là các bước tạo modelcuar CNN thôi.
model.save('tfsign_model/final_model.h5')   

# TEST MODEL

cap = cv2.VideoCapture(0)                                         # Mở camera để test thôi nào
class_name = ['00000','10000','20000','50000']
while(True):
    ret, frame = cap.read()
    frame = cv2.resize(frame, dsize=None,fx=0.5,fy=0.5)           # resize camera (tùy bạn chỉnh)
    # Resize
    image = frame.copy()
    image = cv2.resize(image, dsize=(128, 128))                   # nhớ là size ảnh để test phải bằng size ảnh đã input
    image = image.astype('float')*1./255 
    image = np.expand_dims(image, axis=0)                         # convert về dạng tensor 4 chiều, vì input cũng là 4 chiều
    # Predict
    predict = model.predict(image)                                 # Predict để dự đoán
    #print(predict)
    #print(predict[0])
    #print(class_name[np.argmax(predict)])                         # Những cái này in ra để biết rõ nhé
    #print(np.argmax(predict[0]))
    #print(np.max(predict))
    if np.max(predict)>= 0.8:                                      # Nếu max của predict >= 0.8 thì ta show ảnh
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        color = (0, 255, 0)
        fontScale = 2.5
        cv2.putText(frame, class_name[np.argmax(predict)], org, font,fontScale, color, cv2.LINE_AA)            # Hàm cv2.putText để tạo ra chữ hiện trên camera

    cv2.imshow("Picture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
