import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2

mnist = tf.keras.datasets.mnist
(x_train, y_train) , (x_test, y_test)= mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0
# #print(x_train[0])

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy' , metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.save('digitsRecog.keras')


# model= tf.keras.models.load_model('digitsRecog.keras')
# loss, accuracy= model.evaluate(x_test, y_test)
# print(loss)
# print(accuracy)


n = 0
while os.path.isfile(f"digits/digit{n}.png"):
    try:
        img = cv2.imread(f"digits/digit{n}.png")[:, :, 0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img,  verbose=0)
        print(f"This digit is probably a {np.argmax(prediction)}\n")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        
    except:
        print("Error!")
    finally:
        n += 1
