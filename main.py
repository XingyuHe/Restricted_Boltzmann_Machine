from model import *
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
rbm = RBM(num_visible=784, num_hidden=10,
          train_data=mnist.train.images, epochs=20,
          learning_rate=0.00001)
# rbm.train()

result = np.load("./model/result.npy", allow_pickle=True)

weights, visible_bias, hidden_bias = result[0, 0], result[0, 1], result[0, 2]
orig_img = mnist.test.images[0]
visible = orig_img.reshape(1, -1)
visible = np.random.uniform(1, 255, np.shape(visible)).astype(np.float32)
sample_tensor = rbm.gibbs_sampler(1000, val=visible,
                  hidden_bias=hidden_bias,
                  visible_bias=visible_bias,
                  weights=weights)

sess = tf.Session()
sample = sess.run([sample_tensor])

def show_digit(x: np.array):
    plt.imshow(np.reshape(x, [28, 28]), cmap=plt.cm.gray)
    plt.show()

show_digit(sample)
show_digit(visible)