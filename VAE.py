# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from random import random
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from scipy.stats import norm
from keras.datasets import mnist
from keras import backend as K
from keras.losses import binary_crossentropy
    
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
        
        
# VAE class based on https://keras.io/examples/generative/vae/
class VAE(keras.Model):
    def __init__(self, encoder, decoder, beta, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        #beta-vae will need the beta, as described in the paper https://openreview.net/forum?id=Sy2fzU9gl
        self.beta = beta
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[1]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = encoder(data)
            reconstruction = decoder(z)
            reconstruction_loss = binary_crossentropy(K.flatten(data), K.flatten(reconstruction)) * img_width * img_height
            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var) 
            kl_loss = K.sum(kl_loss, axis = -1)
            kl_loss *= -0.5
            
            # beta to turn this into a beta-VAE
            total_loss = K.mean(reconstruction_loss + self.beta * kl_loss)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": K.mean(kl_loss),
        }
    
def compute_latent(x):
    mu, sigma = x
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    eps = K.random_normal(shape=(batch,dim))
    return mu + K.exp(sigma/2)*eps


# prepare testing data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255
x_test = x_test/255
# as the VAE doesnt need a test set
x_train = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))

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
sigma = layers.Dense(latent_dim)(encoder)
z = layers.Lambda(compute_latent, output_shape=(latent_dim,))([mu, sigma])
conv_shape = K.int_shape(x) # the shape is here
encoder = keras.Model(encoder_inputs, [mu, sigma, z], name = "encoder")
encoder.summary()

latent_inputs = keras.Input(shape = (latent_dim, ))
x = layers.Dense(conv_shape[1]*conv_shape[2]*conv_shape[3], activation = 'relu')(latent_inputs)
x = layers.Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)
x_ = layers.Conv2DTranspose(filters = 16, kernel_size = 3, strides = 2, padding = 'same', activation = 'relu')(x)
x_ = layers.Conv2DTranspose(filters = 8, kernel_size = 3, strides = 2, padding = 'same', activation = 'relu')(x)
x_ =  layers.Conv2DTranspose(filters = num_channels, kernel_size = 3, padding = 'same', activation = 'sigmoid')(x_)
decoder = keras.Model(latent_inputs, x_, name = "decoder")
decoder.summary()

# beta value is input here
vae = VAE(encoder, decoder, beta = 0.1)
vae.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.001))
vae.fit(x_train, epochs = 20, batch_size = 128, verbose = 1)


plt.figure(figsize = (18.64, 9.48))
plt.plot(vae.history.history["loss"], label = "Training Loss")
plt.plot(vae.history.history["kl_loss"], label = "Kullback-Leibler Divergence")
plt.legend()
plt.grid()
plt.show()

# generating data
x_samp = np.copy(x_train)
x_aux = np.copy(x_samp)
# alternatively you can include this step. missing data can be either nan or 0
# x_samp[NanIndex] = 0
x_m, x_l, x_samp = encoder(x_samp)
# z mean and z log var, respectively
x_m = x_m.numpy()
x_l = x_l.numpy()
x_sampled = x_samp.numpy()
# samp contains an example of fully generated data
samp = decoder(x_samp)
samp = samp.numpy()


plot_latent(x_samp, y, latent_dim)
plot_outputs(x_train, samp, 3)

# statistics: histogram and normal distribution to better see the latent space and how it follows a normal distr.
input_x = x_sampled
mean_val = np.mean(input_x)
stddev = np.std(input_x)
domain = np.linspace(np.min(input_x), np.max(input_x))
plt.figure(figsize = (18.64, 9.48))
plt.plot(domain, norm.pdf(domain, mean_val, stddev))
plt.title("Distribution of latent space points with $\mu$ = {:.2f} and $\sigma$ = {:.2f}".format(mean_val, stddev))
plt.xlabel("Value")
plt.ylabel("PDF")
plt.grid()
plt.show()
