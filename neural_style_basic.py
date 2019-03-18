# Scientific Python
import cv2
import numpy as np
# Keras related
from keras import backend as K
from keras.preprocessing.image import img_to_array
from keras.applications.vgg19 import VGG19
# L-BFGS optimizer
from scipy.optimize import fmin_l_bfgs_b
# own modules
from .logging_utils import logger
from .image_utils import compute_gram_matrix, preprocess_image_for_VGG


class NeuralStyleGatys:
    """
    Implements the original Neural Style Transfer algorithm as described in
    'A neural algorithm of artistic style', by Gatys et. al
    """

    def __init__(self, *, style_img_path, content_img_path, output_path, iterations):
        """
        Constructor
        :param style_img_path: input style image path
        :param content_img_path: input content image path
        :param output_path: output image path
        :param iterations: number of iterations used to create the output image
        """
        logger.info("Initialization")
        self.vgg_model = VGG19(weights='imagenet', pooling='avg', include_top='false')
        self.style_img_path = style_img_path
        self.content_img_path = content_img_path
        self.output_path = output_path
        self.iterations = iterations

        # define a dictionary with the layers' name and output
        self.layers_output = dict([(layer.name, layer.output) for layer in self.vgg_model.layers])

        # layers used
        self.content_layer = 'block4_conv2'
        self.style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block4_conv1']

        # input images
        self.content_image = cv2.imread(self.content_img_path)
        (self.img_width, self.img_height, self.img_channels) = self.content_image.shape
        self.style_image = cv2.imread(self.style_img_path)
        self.style_image = cv2.resize(self.style_image, (self.img_height, self.img_width))

        # generated image
        self.output_image = K.placeholder()

        # losses weights
        self.content_loss_weight = 0.001
        self.style_loss_weight = 1

    @staticmethod
    def _calc_content_loss(self, proc_img, input_img):
        """
        Compute the content loss
        :param proc_img:
        :param input_img:
        :return:
        """
        return np.sum(np.square(input_img - proc_img))

    def _calc_style_loss(self, A_out_layers, G_out_layers):
        """
        Compute the style loss
        :param A_out_layers:
        :param G_out_layers:
        """
        assert (A_out_layers.ndim == G_out_layers.ndim and A_out_layers.ndim == 3)
        N = A_out_layers.shape[2]
        M = A_out_layers.shape[0] * A_out_layers.shape[1]
        Gram_A = compute_gram_matrix(A_out_layers)
        Gram_G = compute_gram_matrix(G_out_layers)
        channels = 3
        size = self.img_width * self.img_height
        return K.sum(K.square(Gram_A - Gram_G)) / (4. * (channels ** 2) * (size ** 2))

    def _calc_total_loss(self, c_loss, s_loss, alpha, beta):
        """
        Compute the total loss as a weighted sum of content and style losses
        :param s_loss:
        :param alpha:
        :param beta:
        :return:
        """
        return alpha * c_loss + beta * s_loss

    @staticmethod
    def _calc_style_loss_layer(A, G):
        """
        Compute the style loss for a single layer
        :param A:
        :param G:
        """
        assert (A.ndim == G.ndim and A.ndim == 3)
        N = A.shape[2]
        M = A.shape[0] * A.shape[1]
        Gram_A = compute_gram_matrix(A)
        Gram_G = compute_gram_matrix(G)
