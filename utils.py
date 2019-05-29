
import numpy as np
from scipy import signal
from scipy.misc import imread
from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
import os





def read_image(filename, representation):
    """
    read file and return as matrix. (function from ex1)
    """
    im = imread(filename)
    im = im.astype(np.float64)
    im = im / 255
    if representation == 1:
        im = rgb2gray(im)
    return im



def get_gaussian_filter(filter_size):
    """
    creates 1D Gaussian Kernel of size "kernel size".
    """
    kernel_11 = np.matrix([[1, 1]]).astype(np.float64)
    filter = np.matrix([[1, 1]]).astype(np.float64)
    for i in range(filter_size - 2):
        filter = signal.convolve2d(filter, kernel_11)
    divider = np.sum(filter)
    return filter / divider



def get_kernel_matrix(kernel_size):
    """
    creates Gaussian Kernel of size "kernel size".
    """
    kernel_11 = np.matrix([[1, 1]]).astype(np.float64)
    kernel_1d = np.matrix([[1, 1]]).astype(np.float64)
    for i in range(kernel_size - 2):
        kernel_1d = signal.convolve2d(kernel_1d, kernel_11)
    kernel_1d = kernel_1d.astype(np.float64)
    kernel_1d_T = kernel_1d.T
    kernel_2d = signal.convolve2d(kernel_1d, kernel_1d_T).astype(np.float64)
    return kernel_2d



def reduce(im, filter):
    """
    convolve image on both axis and blur
    """
    blur_x = convolve(im, filter.T)
    blur_x = blur_x[::2]
    blur_x_and_y = convolve(blur_x, filter)
    blur_x_and_y = blur_x_and_y[:, ::2]
    return blur_x_and_y



def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    :param im: input image
    :param max_levels: maximum num of iterations
    :param filter_size: size of gaussian filter
    :return: array of images (the pyramid), the gaussian filter vector
    """
    pyr = [im]
    # create filter
    filter_vec = get_gaussian_filter(filter_size)
    reduced_im = im
    for i in range(1, max_levels):
        reduced_im = reduce(reduced_im, filter_vec)
        if reduced_im.shape[0] < 16 or reduced_im.shape[1] < 16:
            break
        pyr.append(reduced_im)

    return pyr, filter_vec



def expand(im, filter):
    """
    padd image with zeroes, and blur. This creates a larger image.
    """
    new_image_size_tuple = (im.shape[0] * 2, im.shape[1] * 2)
    zero_matrix = np.zeros(new_image_size_tuple).astype(np.float64)
    zero_matrix[::2, ::2] = im
    expand_filter = filter * 2
    blur_x = convolve(zero_matrix, expand_filter)
    blur_x_and_y = convolve(blur_x, expand_filter.T)
    return blur_x_and_y




def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    :param im: input image
    :param max_levels: maximum num of iterations
    :param filter_size: size of gaussian filter
    :return: array of images (the pyramid), the gaussian filter vector
    """
    gaussian_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    pyramid_size = len(gaussian_pyr)
    pyr = []

    for i in range(pyramid_size-1):
        new_lap = gaussian_pyr[i] - expand(gaussian_pyr[i+1], filter_vec)
        pyr.append(new_lap)
    final_lap = gaussian_pyr[ pyramid_size - 1]
    pyr.append(final_lap)
    return pyr, filter_vec



def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    receives a laplacian pyramid, filer vector and coefficients. Returns an image.
    """
    img_with_coeff = lpyr[-1] * coeff[-1]
    for i in range(len(lpyr)-2, -1, -1):
        img_with_coeff = lpyr[i] + expand(img_with_coeff * coeff[i] , filter_vec )
    return img_with_coeff




def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    Main pyramid blending function. Implements the algorithm we learned in class.
    """
    L1, f1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)           # step 1
    L2, f2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)           # step 2
    mask = mask.astype(np.float64)
    Gm, filter_gm = build_gaussian_pyramid(mask, max_levels, filter_size_mask)

    L_out = len(L1) * [None]
    for k in range(len(L1)):
        L_out[k] = Gm[k] * L1[k] + (1-Gm[k]) * L2[k]                            # step 3
    coeff_lst = len(L_out) * [1]
    reconstructed = laplacian_to_image(L_out, f2, coeff_lst)                    # step 4
    return np.clip(reconstructed, 0, 1)



def blur_spatial(im, kernel_size):
    """
    performs image blurring using 2D convolution between the image f and a gaussian
    kernel g.
    :param im: input image to be blurred.
    :param kernel_size: size of the gaussian kernel in each dimension (an odd integer).
    """
    kernel_2d = get_kernel_matrix(kernel_size)
    divider = np.sum(kernel_2d)
    blurred = signal.convolve2d(im, (kernel_2d / divider), mode="same")
    return blurred


