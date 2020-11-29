from model import *
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
train_data = mnist.train.images
train_label = mnist.train.labels
print(np.shape(train_data))
rbm = RBM(num_visible=np.prod(np.shape(train_data)[1:]), num_hidden=10,
          train_data=train_data, epochs=20,
          learning_rate=0.00001)

# rbm.train()

result = np.load("./model/result.npy", allow_pickle=True)

weights, visible_bias, hidden_bias = result[0, 0], result[0, 1], result[0, 2]

test_data = mnist.test.images
test_label = mnist.test.labels

all_numbers = set(list(range(0, 10)))
for i in range(np.shape(test_label)[0]):
    print(i)
    label = np.argmax(test_label[i])
    if label in all_numbers:
        all_numbers.remove(label)
        orig_img = mnist.test.images[i]
        visible = orig_img.reshape(1, -1)


        img_dir = "./pictures/{}".format(label)
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        for j in range(30):
            now = time.time()
            pic_path = "{}/{}.png".format(img_dir, now)
            sample_tensor = rbm.gibbs_sampler(5, val=visible,
                                              hidden_bias=hidden_bias,
                                              visible_bias=visible_bias,
                                              weights=weights,
                                              save_file=True,
                                              img_path=pic_path)


# def show_digit(x: np.array):
#     plt.imshow(np.reshape(x, [28, 28]), cmap=plt.cm.gray)
#     plt.show()
#
# show_digit(sample)
# show_digit(visible)