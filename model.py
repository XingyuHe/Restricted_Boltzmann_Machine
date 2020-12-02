import tensorflow as tf
import numpy as np
import time
import os
import matplotlib.pyplot as plt

class RBM:

    def __init__(self, num_visible, num_hidden, train_data, learning_rate=0.01, epochs=100):
        '''
        :param num_visible: the number of visible nodes
        :param num_hidden: the number of hidden nodes
        :param train_data: the observations
        '''

        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.train_data = train_data
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.build()


    def build(self):
        '''
        Construct the computational graph for tensorflow
        :return: None
        '''
        self.sess = tf.Session()

        self.visible_layer = tf.placeholder(dtype=tf.float32,
                                            shape=[None, self.num_visible],
                                            name="visible_layer")

        self.num_obervation = tf.cast(x=tf.shape(self.visible_layer)[0],
                                      dtype=tf.float32)

        with tf.name_scope("Weights"):
            self.weights = tf.Variable(tf.random_normal(shape=[self.num_visible, self.num_hidden]),
                                 name="weights")
            tf.summary.histogram("weights", self.weights)

        self.visible_bias = tf.Variable(tf.zeros(shape=[1, self.num_visible]),
                                   name="visible_bias")
        self.hidden_bias = tf.Variable(tf.zeros(shape=[1, self.num_hidden]),
                                  name="hidden_bias")

        with tf.name_scope("Reconstruction_error"):
            reconstruction_error = tf.Variable(0.0,name="reconstruction_error")
            tf.summary.scalar('reconstruction_error',reconstruction_error)

        with tf.name_scope("Resconstructio_error"):
            self.reconstruction_error = tf.Variable(0.0,
                                                    name="reconstruction_error",
                                                    dtype=tf.float32)
            tf.summary.scalar('reconstruction_error', self.reconstruction_error)

        self.contrastive_divergence()


    def get_conditional_probabilities(self, layer, weights, val, bias):
        '''
        :param layer: Hidden layer or visible layer, specified as string
        :param weights: Tensorflow placeholder for weight matrix, m x n
        :param val: Input units, hidden or visible as binary or float,

                    If val is value of the visible layer (1 x m), then we are
                    getting the probabilities for the hidden layer. The
                    output dimension should be 1 x n.

                    If val is value of the hidden layer (1 x n), then we are
                    getting the probabilities for the visible layer. The
                    output dimension should be 1 x m

        :param bias: Bias associated with the computation, opposite of the input
        :return: A tensor of probabilities that indicate the probability that the ith hidden node
        is 1 conditioned on the value of the visible layer, or p(H_i == 1 | v) = sigmoid(). The out
        put dimension is
                    1 x n for hidden layer probabilities,
                    1 x m for visible layer probabilities,
        '''

        if layer == "hidden":
            with tf.name_scope(layer):
                return tf.nn.sigmoid(tf.matmul(val, weights) + bias)
        elif layer == "visible":
            with tf.name_scope(layer):
                return tf.nn.sigmoid(tf.matmul(val, tf.transpose(weights)) + bias)
        else:
            print("Wrong layer name! ")


    def contrastive_divergence(self):
        '''
        One step of contrastive divergence and update the gradients to the weights and biases
        :return: None
        '''

        with tf.name_scope("Hidden_probabilities"):
             obs_hid_prob = self.get_conditional_probabilities(layer='hidden', weights=self.weights,
                                                               val=self.visible_layer, bias=self.hidden_bias)

        with tf.name_scope("Positive_Divergence"):
            # the self.visible_layer has a dimension of none x m
            # the obs_hid_prob has a dimension of none x n
            # obs_expectation should have a dimension of  m x n
            obs_expectation = tf.matmul(tf.transpose(self.visible_layer), obs_hid_prob)

        model_hid_states = self.sample(obs_hid_prob)

        model_vis_prob = self.get_conditional_probabilities(layer='visible', weights=self.weights, val=model_hid_states,
                                                            bias=self.visible_bias)

        model_vis_state = tf.floor(obs_hid_prob + tf.random_uniform(tf.shape(obs_hid_prob),
                                                                    0, 1))

        with tf.name_scope("Negative_hidden_probabilities"):
            model_hid_prob = self.get_conditional_probabilities(layer='hidden', weights=self.weights, val=model_vis_prob,
                                                                bias=self.hidden_bias)  # change this to model_vis_state to try
        with tf.name_scope("Negative_Divergence"):
            model_expectation = tf.matmul(tf.transpose(model_vis_prob), model_hid_prob)

        self.divergence_diff = tf.reduce_mean(obs_expectation - model_expectation)
        self.reconstruction_error_ops = tf.reduce_mean(tf.squared_difference(self.visible_layer, model_vis_prob))

        # ==================================== Update the value ==================================
        with tf.name_scope("Weight_gradient"):
            delta_w = tf.multiply(self.learning_rate,
                                  tf.subtract(obs_expectation,
                                              model_expectation))
            weight_gradient_scalar = tf.summary.scalar("weight_increment",
                                                       tf.reduce_sum(delta_w))
        with tf.name_scope("Visible_bias_gradient"):
            delta_visible_bias = tf.multiply(self.learning_rate,
                                             tf.reduce_sum(tf.subtract(self.visible_layer,
                                                                       model_vis_prob),
                                                           axis=0,
                                                           keep_dims=True))

        with tf.name_scope("Hidden_bias_gradient"):
            delta_hidden_bias = tf.multiply(self.learning_rate,
                                            tf.reduce_sum(tf.subtract(obs_hid_prob,
                                                                      model_hid_prob),
                                                          axis=0,
                                                          keep_dims=True))

        self.update = [self.weights.assign_add(delta_w),
                  self.visible_bias.assign_add(delta_visible_bias),
                  self.hidden_bias.assign_add(delta_hidden_bias)]

        self.summary = tf.summary.merge_all()
        summary_path = os.getcwd() + '/RBM_logs/MNIST'
        self.summary_writer = tf.summary.FileWriter(summary_path, self.sess.graph)

    def sample(self, probabilities):

        return tf.floor(probabilities + tf.random_uniform(tf.shape(probabilities),
                                                          0, 1))

    def train(self):
        '''
        This function trains the Boltzmann machine using gradient descent
        :return: result is a 1 x 3 numpy array that contains
                [self.weights, self.visible_bias, self.hidden_bias]
        '''

        self.sess.run(tf.global_variables_initializer())

        start_time = time.time()
        for epoch in range(self.epochs):
            feed_dict = {self.visible_layer: self.train_data}
            result = self.sess.run([self.update], feed_dict=feed_dict)
            errors = self.sess.run([self.divergence_diff, self.reconstruction_error_ops],
                              feed_dict=feed_dict)
            summary = self.sess.run([self.summary], feed_dict=feed_dict)
            self.summary_writer.add_summary(summary[0])
            print("Epoch {}, divergence diff: {}".format(epoch, errors))

        self.result = result
        np.save("./model/result{}".format(start_time), result)
        return result

    def gibbs_sampler(self, steps, val, hidden_bias, visible_bias, weights, save_file=False, img_path=None, return_visible=True):
        '''
        :param steps: The number of samples generated
        :param val: The initial sample
        :param hidden_bias: The bias associated with the hidden layer in the Boltzmann machine
        :param visible_bias: The bias associated with the visible layer in the Boltzmann machine
        :param weights: The weights inside the Botlzmann machine
        :return: The final sample
        '''
        start_time = time.time()
        hidden_p_val = None
        with tf.name_scope("Gibbs_sampling"):
            for i in range(steps):
                hidden_p = self.get_conditional_probabilities(layer="hidden", weights=weights, val=val,
                                                              bias=hidden_bias)

                hidden_p_val = hidden_p
                hidden_sample = self.sample(hidden_p)
                visible_p = self.get_conditional_probabilities(layer="visible", weights=weights, val=hidden_sample,
                                                               bias=visible_bias)
                val = visible_p
                if save_file == True:
                    if img_path == None:
                        pic_dir = os.path.join("./pictures/{}/".format(start_time))
                        if not os.path.exists(pic_dir):
                            os.mkdir(pic_dir)
                        img_path = os.path.join(pic_dir, "{}.png".format(time.time()))

                    plt.imsave(img_path, np.reshape(val.eval(session=self.sess), [28, 28]), cmap=plt.cm.gray)

        if return_visible:
            return val
        else:
            return hidden_p_val











