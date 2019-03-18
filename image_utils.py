#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""A collection of functions used for manipulating and processing images"""

import numpy as np
from keras import backend as K

# CONSTANTS
# mean color Imagenet dataset
MEAN_B = 103.939
MEAN_G = 116.779
MEAN_R = 123.68


def compute_gram_matrix(img):
    """
    Compute the gram matrix of an image, defined as the product of the matrix with its transpose
    :param img: numpy array representing an image
    :return:
    """
    num_dim = img.ndim()
    assert(num_dim == 3 or num_dim == 2)
    if num_dim == 2:
        return img @ img.transpose()
    else:  # num_dim = 3
        flat = K.batch_flatten(np.transpose(img, (2, 0, 1)))
        return flat @ flat.transpose()


def preprocess_image_for_VGG(img):
    """
    Preprocess an image so that it can be fed to a VGG network
    :param img: input image
    :return: preprocessed image
    """
    # Remove mean
    img[:, :, 0] -= MEAN_B
    img[:, :, 1] -= MEAN_G
    img[:, :, 2] -= MEAN_R

    img = img.astype('float32')
    img = np.expand_dims(img, axis=0)

    return img


def postprocess_image_from_VGG(x):
    """
    Convert a tensor to a valid image. Remove preprocessing steps
    :param x: input tensor
    :return: deprocessed image
    """
    (_, w, h, _) = x.shape
    img = x.reshape((w, h, 3))

    img[:, :, 0] += MEAN_B
    img[:, :, 1] += MEAN_G
    img[:, :, 2] += MEAN_R

    img = np.clip(img, 0, 255).astype('uint8')
    return img
