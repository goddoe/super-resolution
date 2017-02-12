import os
import random
import pickle
import math

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt


def random_crop(src, size):
    height = src.shape[0]
    width = src.shape[1]

    max_right = width - size[0] - 1
    max_bottom = height - size[1] - 1

    x = random.randint(0, max_right)
    y = random.randint(0, max_bottom)

    cropped = src[y: y + size[1], x: x + size[0]]

    return cropped


def load_imgs(dir_path, size):
    width = size[0]
    height = size[1]
    name_list = os.listdir(dir_path)
    img_list = []

    for name in name_list:
        img_path = "{}/{}".format(dir_path, name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for i in range(50):
            try:
                img_cropped = random_crop(img, (width, height))
                img_list.append(img_cropped)
            except Exception as e:
                print(e)
    return img_list


def blur_img_list(img_list, size):
    width = size[0]
    height = size[1]

    result_list = []
    for img in img_list:
        w = round(width / 1.5)
        h = round(height / 1.5)
        result = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        result = cv2.resize(result, (width, height), interpolation=cv2.INTER_CUBIC)
        result_list.append(result)
    return result_list


"""
Model
"""


def conv2d(X, n_input, n_output, filter_size, activation=None, name=None):
    with tf.variable_scope(name):
        W = tf.get_variable(
            name='W_1',
            shape=[filter_size[0], filter_size[1], n_input, n_output],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())

        b = tf.get_variable(
            name='b_1',
            shape=[n_output],
            initializer=tf.constant_initializer(0.))

        h = tf.nn.conv2d(X,
                         W,
                         strides=[1, 1, 1, 1],
                         padding='SAME'
                         )
        if activation != None:
            h = activation(tf.nn.bias_add(h, b))

    return h


class USRCNN(object):
    def __init__(self, sess):

        self.sess = sess
        self.shape = (50, 50, 3)
        self.mean_img = None
        self.std_img = None
        self.min_loss = None

        self.build_model()
        sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def load(self, sess, weights_path, meta_path):
        self.saver.restore(sess, weights_path)

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self.mean_img = meta['mean_img']
        self.std_img = meta['std_img']
        self.min_loss = meta['min_loss']

    def save(self, sess, weights_path, meta_path, min_loss, flag_export_graph=False, graph_path=None):
        meta = {
            "mean_img": self.mean_img,
            "std_img": self.std_img,
            "shape": self.shape,
            "min_loss": min_loss
        }

        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

        self.saver.save(sess, weights_path, latest_filename="recent.ckpt", write_meta_graph=flag_export_graph)

    def build_model(self):
        height = self.shape[0]
        width = self.shape[1]
        channel = self.shape[2]

        n_flat = height * width * channel
        X = tf.placeholder(tf.float32, shape=[None, height, width, channel], name='X')
        Y = tf.placeholder(tf.float32, shape=[None, height, width, channel], name='Y')

        start_learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        global_step = tf.Variable(0, trainable=False)

        layer_info_list = [
            {'name': 'conv_1',
             'n_input': 3,
             'n_output': 64,
             'filter_size': (3, 3),
             'activation': tf.nn.relu},
            {'name': 'conv_2',
             'n_input': 64,
             'n_output': 64,
             'filter_size': (3, 3),
             'activation': tf.nn.relu},
            {'name': 'conv_3',
             'n_input': 64,
             'n_output': 64,
             'filter_size': (3, 3),
             'activation': tf.nn.relu},
            {'name': 'conv_4',
             'n_input': 64,
             'n_output': 64,
             'filter_size': (3, 3),
             'activation': tf.nn.relu},
            {'name': 'conv_5',
             'n_input': 64,
             'n_output': 3,
             'filter_size': (3, 3),
             'activation': None},
        ]

        current_input = X
        for info in layer_info_list:
            current_input = conv2d(X=current_input,
                                   n_input=info['n_input'],
                                   n_output=info['n_output'],
                                   filter_size=info['filter_size'],
                                   activation=info['activation'],
                                   name=info['name'],
                                   )

        Y_pred = current_input + X

        Y_pred_flattened = tf.reshape(Y_pred, shape=[-1, n_flat])
        Y_flattened = tf.reshape(Y, shape=[-1, n_flat])

        learning_rate = tf.train.exponential_decay(start_learning_rate, global_step,
                                                   1000, 0.96, staircase=True)

        cost = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(Y_pred_flattened, Y_flattened), axis=1), axis=0)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        self.X = X
        self.Y = Y
        self.Y_pred = Y_pred
        self.cost = cost
        self.optimizer = optimizer
        self.start_learning_rate = start_learning_rate
        self.gloabal_step = global_step

    def train(self, X_train, Y_train, batch_size, n_epoch,start_learning_rate, save_dir_path, X_valid=None, Y_valid=None):

        fig, axs = plt.subplots(1, 4, figsize=(20, 6))
        if self.min_loss is None:
            self.min_loss = 999999999
        # figure
        epoch_list = []
        loss_list = []

        if self.mean_img is None or self.std_img is None:
            self.mean_img = np.mean(Y_train, axis=0)
            self.std_img = np.std(Y_train, axis=0)
            print("make mean_img and std_img")

        height = self.shape[0]
        width = self.shape[1]
        channel = self.shape[2]

        test_img_source = (X_train, Y_train) if Y_valid is None else (X_valid, Y_valid)
        test_img_idx = random.randint(0, len(test_img_source))

        for epoch_i in range(n_epoch):
            print("epoh_i : {}".format(epoch_i))
            rand_idx_list = np.random.permutation(range(len(X_train)))
            n_batch = len(rand_idx_list) // batch_size
            for batch_i in range(n_batch):
                rand_idx = rand_idx_list[batch_i * batch_size: (batch_i + 1) * batch_size]
                batch_x = X_train[rand_idx]
                batch_y = Y_train[rand_idx]
                self.sess.run(self.optimizer,
                              feed_dict={self.X: (batch_x - self.mean_img) / self.std_img,
                                         self.Y: (batch_y - self.mean_img) / self.std_img,
                                         self.start_learning_rate: start_learning_rate})

            loss = self.sess.run(self.cost, feed_dict={self.X: (X_valid - self.mean_img) / self.std_img,
                                                       self.Y: (Y_valid - self.mean_img) / self.std_img})
            print("loss : {}".format(loss))
            epoch_list.append(epoch_i)
            loss_list.append(loss)

            if loss < self.min_loss:
                self.min_loss = loss
                weights_path = "{}/weights".format(save_dir_path)
                meta_path = "{}/meta_data.pickle".format(save_dir_path)
                self.save(self.sess, weights_path=weights_path, meta_path=meta_path, min_loss=self.min_loss)

                print("-" * 30)
                print("Saved!")
                print("weights_path : {}".format(weights_path))
                print("meta_data_path : {}".format(meta_path))
                print("-" * 30)


            if epoch_i % 10 == 0:
                test_img_origin = test_img_source[1][test_img_idx]
                test_img_query = test_img_source[0][test_img_idx]
                test_img_recon = np.reshape(test_img_query, [-1, height, width, channel])
                test_img_recon = self.Y_pred.eval(feed_dict={self.X: (test_img_recon - self.mean_img) / self.std_img},
                                                  session=self.sess)
                test_img_recon = np.reshape(test_img_recon, [height, width, channel])
                test_img_recon = test_img_recon * self.std_img + self.mean_img
                test_img_recon = np.clip(test_img_recon, 0, 255)
                test_img_recon = test_img_recon.astype(np.uint8)

                axs[0].imshow(test_img_origin)
                axs[0].set_title("origin")
                axs[1].imshow(test_img_query)
                axs[1].set_title("query")
                axs[2].imshow(test_img_recon)
                axs[2].set_title("reconstructed image_{}".format(epoch_i))
                axs[3].plot(epoch_list, loss_list)
                axs[3].set_xlabel("epoch_i")
                axs[3].set_ylabel("loss")
                axs[3].set_title("loss_{}".format(epoch_i))
                plt.pause(0.05)

        return self.sess

    def run(self, src):

        expand_ratio = 1.2
        times = 8

        target = src
        target = self.enhance_resolution(target)

        for i in range(times):
            shape = target.shape

            height_resize = round(shape[0] * expand_ratio)
            width_resize = round(shape[1] * expand_ratio)

            target = cv2.resize(target, (width_resize, height_resize))
            target = self.enhance_resolution(target)

        return target

    def enhance_resolution(self, src):

        height = self.shape[0]
        width = self.shape[1]
        channel = self.shape[2]

        mean_img = self.mean_img
        std_img = self.std_img

        patch_list, shape = self.divide_img_to_patch(src, (width, height))

        patch_recon_list = []

        for patch in patch_list:
            patch_normalized = (patch - mean_img) / std_img
            patch_normalized = patch_normalized.reshape([1, height, width, channel])
            patch_recon = self.sess.run(self.Y_pred, feed_dict={self.X: patch_normalized})
            patch_recon = np.reshape(patch_recon, [height, width, channel])
            patch_recon = patch_recon * std_img + mean_img
            patch_recon = np.clip(patch_recon, 0, 255)
            patch_recon = patch_recon.astype(np.uint8)
            patch_recon_list.append(patch_recon)

        row_list = []
        for row in range(shape[0]):
            col_list = []
            for col in range(shape[1]):
                col_list.append(patch_recon_list[row * shape[1] + col])
            row = np.concatenate(col_list, axis=1)
            row_list.append(row)

        recon_img = np.concatenate(row_list, axis=0)
        recon_img = recon_img[:src.shape[0], :src.shape[1]]

        return recon_img

    def divide_img_to_patch(self, src, size):
        patch_list = []

        img_h = src.shape[0]
        img_w = src.shape[1]

        size_h = size[1]
        size_w = size[0]

        width_q = math.ceil(img_w / size_w)
        height_q = math.ceil(img_h / size_h)

        background = np.zeros(shape=(height_q * size_h, width_q * size_w, 3), dtype=src.dtype)
        background[:img_h, :img_w] = src
        src_with_background = background

        shape = (height_q, width_q)

        for h_i in range(height_q):
            for w_i in range(width_q):
                patch = src_with_background[h_i * size_h:(h_i + 1) * size_h, w_i * size_w: (w_i + 1) * size_w]
                patch_list.append(patch)

        return patch_list, shape


def run():
    height = 50
    width = 50
    channel = 3

    img_list = load_imgs("./data/urban_hr", (width, height))

    X_all = np.array(blur_img_list(img_list, (width, height)))
    Y_all = np.array(img_list)

    mean_img = np.mean(Y_all, axis=0)
    std_img = np.std(Y_all, axis=0)

    # data
    rand_idx = np.random.permutation(range(len(X_all)))
    X_all = X_all[rand_idx]
    Y_all = Y_all[rand_idx]

    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1

    data_num = len(X_all)
    train_data_num = round(data_num * train_ratio)
    valid_data_num = round(data_num * valid_ratio)
    test_data_num = round(data_num * test_ratio)

    X_train = X_all[:train_data_num]
    Y_train = Y_all[:train_data_num]
    X_valid = X_all[train_data_num:train_data_num + valid_data_num]
    Y_valid = Y_all[train_data_num:train_data_num + valid_data_num]
    X_test = X_all[train_data_num + valid_data_num:train_data_num + valid_data_num + test_data_num]
    Y_test = Y_all[train_data_num + valid_data_num:train_data_num + valid_data_num + test_data_num]

    sess = tf.Session()
    usrcnn = USRCNN(sess)
    usrcnn.load(sess,'./model/weights','./model/meta_data.pickle')
    usrcnn.train(X_train, Y_train, X_valid=X_valid, Y_valid=Y_valid,
                 batch_size=64, n_epoch=3000, save_dir_path='./model')

    return usrcnn


def test():
    def load_imgs(dir_path):
        name_list = os.listdir(dir_path)

        img_list = []

        for name in name_list:
            img_path = "{}/{}".format(dir_path, name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_list.append(img)
        return img_list

    test_img_list = load_imgs("./data/celeba")
    test_img = test_img_list[10]
    test_img_resized = cv2.resize(test_img, (test_img.shape[1] // 3, test_img.shape[0] // 3),
                                  interpolation=cv2.INTER_CUBIC)
    test_img_resized = cv2.resize(test_img_resized, (test_img.shape[1], test_img.shape[0]),
                                  interpolation=cv2.INTER_CUBIC)

    with tf.Session() as sess:
        usrcnn = USRCNN(sess)
        usrcnn.load(sess, './model/weights', "./model/meta_data.pickle")
        result = usrcnn.enhance_resolution(test_img_resized)
        plt.imshow(result)


"""

def load_imgs(dir_path):
    name_list = os.listdir(dir_path)

    img_list = []

    for name in name_list:
        img_path = "{}/{}".format(dir_path, name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_list.append(img)
    return img_list
test_img_list = load_imgs("./data/celeba")
test_img = test_img_list[10]
test_img_resized = cv2.resize(test_img, (test_img.shape[1]//3, test_img.shape[0]//3), interpolation = cv2.INTER_CUBIC)
test_img_resized = cv2.resize(test_img_resized, (test_img.shape[1],test_img.shape[0]), interpolation = cv2.INTER_CUBIC)


sess = tf.Session():
usrcnn = USRCNN(sess)
usrcnn.load(sess, './model/weights', "./model/meta_data.pickle")
result = usrcnn.enhance_resolution(test_img_resized)
plt.imshow(result)

"""
