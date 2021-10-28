# -*- coding: utf-8 -*-
import numpy as np
from random import random
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
import keras.backend as K
from keras.losses import binary_crossentropy
    

def plot_outputs(orig, samples, pictures):   
    fig, axs = plt.subplots(pictures, 2)
    for i in range(pictures):
        rng = int(samples.shape[0]*random()) # random number
    
        rec_image = np.array(orig[rng], dtype='float')
        rec_pixels = rec_image.reshape((28, 28))
        axs[i,0].imshow(rec_pixels, cmap='gray')
        
        orig_image = np.array(samples[rng], dtype='float')
        orig_pixels = orig_image.reshape((28, 28))
        axs[i,1].imshow(orig_pixels, cmap='gray')



# prepare testing data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255
x_test = x_test/255


# introducing noise
noise = layers.GaussianNoise(0.5)
noisy = noise(x_train.astype(np.float32), training=True)   
x_train = noisy.numpy()

noise = layers.GaussianNoise(0.5)
noisy = noise(x_test.astype(np.float32), training=True)   
x_test = noisy.numpy()

# clipping is a must to keep the crossentropy working as it should!
x_train = np.clip(x_train, 0, 1)
x_test = np.clip(x_test, 0, 1)


# Convert from (#, 28, 28) to (#, 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

img_height = x_train.shape[1]
img_width = x_train.shape[2]
num_channels = x_train.shape[3]

# layers for both the encoder and the decoder based on 
# https://becominghuman.ai/using-variational-autoencoder-vae-to-generate-new-images-14328877e88d
# https://blog.keras.io/building-autoencoders-in-keras.html
# https://keras.io/examples/vision/autoencoder/

AE = Sequential()
encoder_inputs = keras.Input(shape = (img_height, img_width, num_channels))
# encoder
AE.add(encoder_inputs)
AE.add(layers.Conv2D(16, (3, 3), padding = 'same', activation = 'relu'))
AE.add(layers.MaxPooling2D((2, 2), padding = 'same'))
AE.add(layers.Conv2D(16, (3, 3),  padding = 'same', activation = 'relu'))
AE.add(layers.MaxPooling2D((2, 2), padding = 'same'))

# decoder
AE.add(layers.Conv2DTranspose(16, (3, 3), strides = 2, activation = 'relu', padding = 'same'))
AE.add(layers.Conv2DTranspose(16, (3, 3), strides = 2, activation = 'relu', padding = 'same'))
AE.add(layers.Conv2D(num_channels, (3, 3), activation = 'sigmoid', padding = 'same'))


def loss_func(data, pred):
    return K.mean(binary_crossentropy(K.flatten(data), K.flatten(pred)) * img_width * img_height)

AE.compile(optimizer = 'adam', loss = loss_func)
AE.summary()
history = AE.fit(x_train, x_train, epochs = 20, batch_size = 128, verbose = 1, validation_split = 0.1)

plt.figure(figsize = (18.64, 9.48))
plt.plot(history.history["loss"], label = "Training Loss")
plt.plot(history.history["val_loss"], label = "Validation Loss")
plt.legend()
plt.grid()
plt.show()

preds = AE.predict(x_test)
plot_outputs(x_test, preds, 3)


# saving the predictions to test in the classification model
with open ('denoised.npy', 'wb') as f:
    np.save(f, preds)