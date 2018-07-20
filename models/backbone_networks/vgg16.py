"""
Implementation of the VGG16 backbone network.

Based on:
https://arxiv.org/abs/1409.1556
Structure in order to be able to re-use weights from:
http://www.cs.toronto.edu/~frossard/post/vgg16/
"""

import numpy as np
import tensorflow as tf


class VGG16(object):

    def __init__(self, weights: str, mean: list=[123.68, 116.779, 103.939]):
        """

        :param weights: Path to the weights (for ImageNet downloaded from
            http://www.cs.toronto.edu/~frossard/post/vgg16)
        :param mean: Mean used for the training.
        """
        self.mean = mean
        self.weights = np.load(weights)

    def vgg16_backbone(self, net):
        # substract dataset mean
        mean = tf.constant(self.mean, dtype=tf.float32, shape=[1, 1, 1, 3],
                           name='images_mean')
        net = net - mean

        with tf.variable_scope('vgg16', reuse=tf.AUTO_REUSE):
            conv1_1 = tf.layers.Conv2D(
                64, 3, padding='same', activation='relu', trainable=False,
                kernel_initializer=tf.constant_initializer(
                    self.weights['conv1_1_W']),
                bias_initializer=tf.constant_initializer(
                    self.weights['conv1_1_b']))(net)

            conv1_2 = tf.layers.Conv2D(
                64, 3, padding='same', activation='relu', trainable=False,
                kernel_initializer=tf.constant_initializer(
                    self.weights['conv1_2_W']),
                bias_initializer=tf.constant_initializer(
                    self.weights['conv1_2_b']))(conv1_1)

            pool1 = tf.layers.MaxPooling2D(2, 2, padding='same')(conv1_2)

            conv2_1 = tf.layers.Conv2D(
                128, 3, padding='same', activation='relu', trainable=False,
                kernel_initializer=tf.constant_initializer(
                    self.weights['conv2_1_W']),
                bias_initializer=tf.constant_initializer(
                    self.weights['conv2_1_b']))(pool1)

            conv2_2 = tf.layers.Conv2D(
                128, 3, padding='same', activation='relu', trainable=False,
                kernel_initializer=tf.constant_initializer(
                    self.weights['conv2_2_W']),
                bias_initializer=tf.constant_initializer(
                    self.weights['conv2_2_b']))(conv2_1)

            pool2 = tf.layers.MaxPooling2D(2, 2, padding='same')(conv2_2)

            conv3_1 = tf.layers.Conv2D(
                256, 3, padding='same', activation='relu', trainable=False,
                kernel_initializer=tf.constant_initializer(
                    self.weights['conv3_1_W']),
                bias_initializer=tf.constant_initializer(
                    self.weights['conv3_1_b']))(pool2)

            conv3_2 = tf.layers.Conv2D(
                256, 3, padding='same', activation='relu', trainable=False,
                kernel_initializer=tf.constant_initializer(
                    self.weights['conv3_2_W']),
                bias_initializer=tf.constant_initializer(
                    self.weights['conv3_2_b']))(conv3_1)

            conv3_3 = tf.layers.Conv2D(
                256, 3, padding='same', activation='relu', trainable=False,
                kernel_initializer=tf.constant_initializer(
                    self.weights['conv3_3_W']),
                bias_initializer=tf.constant_initializer(
                    self.weights['conv3_3_b']))(conv3_2)

            pool3 = tf.layers.MaxPooling2D(2, 2, padding='same')(conv3_3)

            conv4_1 = tf.layers.Conv2D(
                512, 3, padding='same', activation='relu', trainable=False,
                kernel_initializer=tf.constant_initializer(
                    self.weights['conv4_1_W']),
                bias_initializer=tf.constant_initializer(
                    self.weights['conv4_1_b']))(pool3)

            conv4_2 = tf.layers.Conv2D(
                512, 3, padding='same', activation='relu', trainable=False,
                kernel_initializer=tf.constant_initializer(
                    self.weights['conv4_2_W']),
                bias_initializer=tf.constant_initializer(
                    self.weights['conv4_2_b']))(conv4_1)

            conv4_3 = tf.layers.Conv2D(
                512, 3, padding='same', activation='relu', trainable=False,
                kernel_initializer=tf.constant_initializer(
                    self.weights['conv4_3_W']),
                bias_initializer=tf.constant_initializer(
                    self.weights['conv4_3_b']))(conv4_2)

        return conv1_2, conv2_2, conv3_3, conv4_3
