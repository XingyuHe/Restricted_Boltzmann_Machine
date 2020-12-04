from model import *
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
# from plot import  *


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
train_data = mnist.train.images
train_label = mnist.train.labels
print(np.shape(train_data))
rbm = RBM(num_visible=np.prod(np.shape(train_data)[1:]), num_hidden=10,
          train_data=train_data, epochs=500,
          learning_rate=0.00001)

rbm.train()

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

# What do the hidden unit look like?
all_hidden_val = {} # 10 x num_hidden variables
all_hidden_vec = np.zeros([10, 10])
all_hidden_visit = {}
test_data = mnist.test.images
test_label = mnist.test.labels
all_numbers = set(list(range(0, 10)))
result = np.load("./model/result-[728, 100].npy", allow_pickle=True)
weights, visible_bias, hidden_bias = result[0, 0], result[0, 1], result[0, 2]

# for i in range(np.shape(test_label)[0]):
for i in range(1000):
    print(i)
    label = np.argmax(test_label[i])
    if label in all_numbers:
        # all_numbers.remove(label)
        orig_img = mnist.test.images[i]
        visible = orig_img.reshape(1, -1)

        hidden_prob = rbm.gibbs_sampler(1, val=visible,
                                          hidden_bias=hidden_bias,
                                          visible_bias=visible_bias,
                                          weights=weights, return_visible=False)
        sess=tf.Session()
        hidden_prob_val = hidden_prob.eval(session=sess)

        if label not in all_hidden_val:
            all_hidden_val[label] = hidden_prob_val[0]
            all_hidden_visit[label] = 1
        else:
            all_hidden_visit[label] += 1
            all_hidden_val[label] += hidden_prob_val[0]
            all_hidden_vec[label, :] = all_hidden_val[label] / all_hidden_visit[label]

    if i % 10 == 0 and i != 0:
        plt.imshow(all_hidden_vec, cmap="jet")
        plt.colorbar()
        plt.xlabel("Hidden Units Values")
        plt.ylabel("The Corresponding Number in the Sampled Visible Layer")
        plt.title("10 hidden units, round {}".format(i))
        plt.savefig("./pictures/hidden_layer_values_[728, 10]/hidden_layer_value_[728, 10]_round_{}".format(i))
        plt.close()

for label in range(10):
    raw_val = all_hidden_val[label]
    print(raw_val)
    norm_val = (raw_val - np.average(raw_val))/ np.std(raw_val)
    all_hidden_vec[label] =norm_val

plt.imshow(all_hidden_vec, cmap="jet")
plt.colorbar()
plt.xlabel("Hidden Units Values")
plt.ylabel("The Corresponding Number in the Sampled Visible Layer")
plt.title("Feature Extraction on MNIST using a Restricted Boltzmann Machine with 100 hidden units")


# plt.annotate("Each horizontal vector represents the feature of the images associated with pictures with digit y ",
#              (0.5,0.5))

# plot a visible sample
orig_img = mnist.test.images[-1000]
visible = orig_img.reshape(1, -1)
visible_sample = rbm.gibbs_sampler(1, val=visible,
                  hidden_bias=hidden_bias,
                  visible_bias=visible_bias,
                  weights=weights)

sess=tf.Session()
visible_sample = visible_sample.eval(session=sess)
visible_sample = np.reshape(visible_sample, [28, 28])

plt.imshow(visible_sample, cmap=plt.cm.gray)
plt.imshow(np.reshape(orig_img, [28, 28]), cmap=plt.cm.gray)

# return tf.floor(probabilities + tf.random_uniform(tf.shape(probabilities),
#                                                   0, 1))
os.mkdir("./pictures/7_sample")
for i in range(40):
    sample = np.floor(visible_sample + np.random.uniform(0, 1, np.shape(visible_sample)))
    plt.imsave("./pictures/7_sample/{}.png".format(i), sample, cmap=plt.cm.gray, dpi=100000)










# def show_digit(x: np.array):
#     plt.imshow(np.reshape(x, [28, 28]), cmap=plt.cm.gray)
#     plt.show()
#
# show_digit(sample)
# show_digit(visible)