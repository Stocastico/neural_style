# Scientific Python
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Keras related
from keras import backend as K
from keras.preprocessing.image import img_to_array
from keras.applications.vgg19 import VGG19

vgg_model = VGG19(weights='imagenet', pooling='avg', include_top='false')

# define a dictionary with the layers' name and output
layers_output = dict([(layer.name, layer.output) for layer in vgg_model.layers])

#images path
content_img_path = './content_images/donostia.jpg'
style_img_path = './style_images/starry_night.jpg'


content_layer = 'block4_conv2'
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block4_conv1']

content_loss_weight = 1
style_loss_weight = 1000

#helper functions
def calc_content_loss(input_img, proc_img):
    return np.sum(np.square(input_img - proc_img))


def calc_style_loss_layer(A, G):
    assert (A.ndim == G.ndim and A.ndim == 3)
    N = A.shape[2]
    M = A.shape[0] * A.shape[1]
    Gram_A = compute_gram_matrix(A)
    Gram_G = compute_gram_matrix(G)


def calc_style_loss(A_out_layers, G_out_layers):
    pass


def calc_total_loss(c_loss, s_loss, alfa, beta):
    return alfa * c_loss + beta * s_loss


def compute_gram_matrix(img):
    pass


def preprocess_image(img):
    pass

# as we load image with opencv, there is no need to convert to BGR
# we only have to remove the mean

def read_image_from_file(img_path):
    img = cv2.imread(img_path)
    img = img_to_array(img)