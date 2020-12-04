from model import *
import matplotlib.pyplot as plt
import numpy as np
import imageio
from tensorflow.examples.tutorials.mnist import input_data




def compile_images():
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
            for j in range(20):
                now = time.time()
                pic_path = "{}/{}.png".format(img_dir, now)
                sample_tensor = rbm.gibbs_sampler(10, val=visible,
                                                  hidden_bias=hidden_bias,
                                                  visible_bias=visible_bias,
                                                  weights=weights,
                                                  save_file=True,
                                                  img_path=pic_path)


def create_gif_state_value_function(sourceFileNames, targetFileName):
    images = []
    for filename in sourceFileNames:
        images.append(imageio.imread(filename, quality=10))
    imageio.mimsave('{}.gif'.format(targetFileName), images)

dir = "./pictures/7_sample"
filenames = os.listdir(dir)

for j in range(len(filenames)):
    filenames[j] = "{}/{}".format(dir, filenames[j])
    #         imgs[j] = "./pictures/{}/{}".format(i, imgs[j])
filenames.sort(key=os.path.getctime)

create_gif_state_value_function(filenames, "./pictures/7_sample.gif")
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# train_data = mnist.train.images
# train_label = mnist.train.labels
# print(np.shape(train_data))
# rbm = RBM(num_visible=np.prod(np.shape(train_data)[1:]), num_hidden=10,
#           train_data=train_data, epochs=20,
#           learning_rate=0.00001)
# # rbm.train()
#
# result = np.load("./model/result.npy", allow_pickle=True)
#
# for i in range(10):
#     imgs = os.listdir("./pictures/{}/".format(i))
#     for j in range(len(imgs)):
#         imgs[j] = "./pictures/{}/{}".format(i, imgs[j])
#     print(imgs)
#     create_gif_state_value_function(list(imgs), "./pictures/{}".format(i))
#
#
