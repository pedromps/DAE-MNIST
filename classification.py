# -*- coding: utf-8 -*-
from tensorflow.keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GaussianNoise
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# one-hot encoding
y_train = to_categorical(y_train, num_classes = 10)
y_test = to_categorical(y_test, num_classes = 10)

# normalising pixels
x_train = np.array(x_train, dtype = 'float64')
x_test = np.array(x_test, dtype = 'float64')
x_train /= 255
x_test /= 255


with open('denoised.npy', 'rb') as f:
    x_test_denoised = np.load(f)
    
noise = GaussianNoise(0.5)
noisy = noise(x_test.astype(np.float32), training=True)   
x_test_noisy = noisy.numpy()
x_test_noisy = np.clip(x_test_noisy, 0, 1)
 
# reshape
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
x_test_noisy = x_test_noisy.reshape((x_test_noisy.shape[0], 28, 28, 1))
x_test_denoised = x_test_denoised.reshape((x_test_denoised.shape[0], 28, 28, 1))

# I used this mode in a previous assignment in University
model = Sequential()
model.add(Conv2D(16, (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# fitting the model and then plotting it
history = model.fit(x_train, y_train, batch_size = 32, epochs = 10, validation_split = 0.1)
plt.figure()
plt.plot(history.history['loss'])
plt.grid()

# predictions
y_pred = np.argmax(model.predict(x_test), axis = -1)
y_pred_noisy = np.argmax(model.predict(x_test_noisy), axis = -1)
y_pred_denoised = np.argmax(model.predict(x_test_denoised), axis = -1)

# metrics calculated here
cm = confusion_matrix(np.argmax(y_test, axis = 1), y_pred)
acc = 100*accuracy_score(y_test, to_categorical(y_pred, num_classes = 10))

# metrics calculated here
cm_noisy = confusion_matrix(np.argmax(y_test, axis = 1), y_pred_noisy)
acc_noisy = 100*accuracy_score(y_test, to_categorical(y_pred_noisy, num_classes = 10))

# metrics calculated here
cm_denoised = confusion_matrix(np.argmax(y_test, axis = 1), y_pred_denoised)
acc_denoised = 100*accuracy_score(y_test, to_categorical(y_pred_denoised, num_classes = 10))
