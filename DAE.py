# -*- coding: utf-8 -*-
import numpy as np
from random import random
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras import backend as K

def plot_latent(x, y, latent_dim):
    if latent_dim != 2:
        return
    else:
        # latent space plot
        plt.figure(figsize=(14,12))
        plt.scatter(x[:,0], x[:,1], s=2, c=y, cmap='hsv')
        plt.colorbar()
        plt.grid()
        plt.show()
    
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
    
# as the VAE doesnt need a test set
x_train = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))


# introducing noise
for i in range(x_train.shape[0]):
    x_train[i] += np.random.uniform(0, 1, ((28, 28)))


# Convert from (#, 28, 28) to (#, 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)

img_height = x_train.shape[1]
img_width = x_train.shape[2]
num_channels = x_train.shape[3]
latent_dim = 6 # it makes sense to plot when latent_dim is 2



# layers for both the encoder and the decoder based on https://becominghuman.ai/using-variational-autoencoder-vae-to-generate-new-images-14328877e88d
encoder_inputs = keras.Input(shape = (img_height, img_width, num_channels))
x = layers.Conv2D(filters = 8, kernel_size = 3, strides = 2, padding = 'same', activation = 'relu')(encoder_inputs)
x = layers.Conv2D(filters = 16, kernel_size = 3, strides = 2,  padding = 'same', activation = 'relu')(encoder_inputs)
encoder = layers.Flatten()(x)
mu = layers.Dense(latent_dim)(encoder)
conv_shape = K.int_shape(x) # the shape is here
x = layers.Dense(conv_shape[1]*conv_shape[2]*conv_shape[3], activation = 'relu')(mu)
x = layers.Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)
x_ = layers.Conv2DTranspose(filters = 16, kernel_size = 3, strides = 2, padding = 'same', activation = 'relu')(x)
x_ = layers.Conv2DTranspose(filters = 8, kernel_size = 3, strides = 2, padding = 'same', activation = 'relu')(x)
x_ =  layers.Conv2DTranspose(filters = num_channels, kernel_size = 3, padding = 'same', activation = 'sigmoid')(x_)
model = keras.Model(encoder_inputs, x_, name = "autoencoder")

# beta value is input here
model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
model.summary()
model.fit(x_train, epochs = 20, batch_size = 128, verbose = 1)

