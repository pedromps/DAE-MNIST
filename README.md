# DAE and VAE 
Denoising with an Autoencoder to see if it improves classification for MNIST data

An Autoencoder was trained with training set of the MNIST data with Gaussian Noise added to it, with the goal of denoising the initial data in the output. In a following step, this AE would receive as input the test data of the MNIST dataset and output denoised data. This latter was saved and later used in the classification task.


For the classification task, a CNN is trained with the normal MNIST data for the classification problem. Then, using clean test data as input, a benchmark was established, which achieved around 98% accuracy. The noisy test data (made by adding Gaussian Noise to the previous test data) was used as input and had an accuracy of around 63%. The denoised data produced an accuracy of 81%. 


Thus it is visible that the Denoising Autoencoder worked as intended, as it removed enough noise from the noisy data to the point that the classification output much better results.
