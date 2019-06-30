import datetime
import os
import random as rn

import barcode
import cv2
import numpy as np
from barcode.writer import ImageWriter
from matplotlib import pyplot as plt

import settings


def rotate3DImage(input, alpha, beta, gamma, dx, dy, dz, f):
    """
    coordinate system:
    y
    ^
    |
    |
    |z(pointing inwards)--------->x
    :param input: image to rotate
    :param alpha: rotation angle around x-axis
    :param beta: rotation angle around y-axis
    :param gamma: rotation angle around z-axis
    :param dx: translation in x-axis
    :param dy: translation in y-axis
    :param dz: translation in z-axis (recommended value: 200)
    :param f: focal distance of camera (recommended value: 200)
    :return: rotated, moved image
    """
    # convert to radians
    alpha = alpha*np.pi/180.0
    beta = beta*np.pi/180.0
    gamma = gamma*np.pi/180.0
    rows, cols, ch = input.shape
    w = cols
    h = rows
    # projection 2D-->3D
    A1 = [[1, 0, -w/2], [0, 1, -h/2], [0, 0, 0], [0, 0, 1]]
    # rotation matrices
    Rx = [[1, 0, 0, 0], [0, np.cos(alpha), -np.sin(alpha), 0], [0, np.sin(alpha), np.cos(alpha), 0], [0, 0, 0, 1]]
    Ry = [[np.cos(beta), 0, -np.sin(beta), 0], [0, 1, 0, 0], [np.sin(beta), 0, np.cos(beta), 0], [0, 0, 0, 1]]
    Rz = [[np.cos(gamma), -np.sin(gamma), 0, 0], [np.sin(gamma), np.cos(gamma), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    # composed rotation matrix
    tmp = np.dot(Rx, Ry)
    R = np.dot(tmp, Rz)
    # translation matrix
    T = [[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz], [0, 0, 0, 1]]
    # 3D-->2D matrix
    A2 = [[f, 0, w/2, 0], [0, f, h/2, 0], [0, 0, 1, 0]]
    # final transformation matrix
    tmp1 = np.dot(R, A1)
    tmp2 = np.dot(T, tmp1)
    trans = np.dot(A2, tmp2)
    dst = cv2.warpPerspective(img, trans, (cols, rows), cv2.INTER_LANCZOS4)
    return dst


def change_contrast_brightness(img, alpha, beta):
    """
    changes the image with: alpha(pixel_value) + beta
    alpha 1  beta 0      --> no change
    0 < alpha < 1        --> lower contrast
    alpha > 1            --> higher contrast
    -127 < beta < +127   --> good range for brightness values
    :param img: image to change contrast and brightness
    :param alpha: contrast factor
    :param beta: value of brightness change
    :return: image with changed contrast and brightness
    """
    return cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)


def random_with_n_digits(n):
    # TODO: make sure radint has even distribution of numbers
    """
    :param n:
    :return: string with n-digit number including 0 in the first digit e.g. 0142
    """
    stringList = []
    for x in range(n):
        stringList.append(str(rn.randint(0, 9)))
    string = ''.join(stringList)
    return string


def demo():
    img = cv2.imread('./output/example.png')
    # blur
    blur = cv2.blur(img, (3, 3))
    # gaussian noise
    row, col, ch = img.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    gauss_noise = img + gauss
    # salt and pepper noise
    row, col, ch = img.shape
    s_vs_p = 0.5
    amount = 0.004
    out = np.copy(img)
    # Salt mode
    num_salt = np.ceil(amount * img.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    out[coords] = 1
    # Pepper mode
    num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    out[coords] = 0
    # poisson noise
    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))
    poisson = np.random.poisson(img * vals) / float(vals)
    # speckle noise
    row, col, ch = img.shape
    gauss = np.random.randn(row, col, ch)
    gauss = gauss.reshape(row, col, ch)
    speckle = img + img * gauss
    plt.subplot(321), plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(322), plt.imshow(blur),plt.title('Blurred')
    plt.xticks([]), plt.yticks([])
    plt.subplot(323), plt.imshow(gauss_noise),plt.title('Gaussian Noise')
    plt.xticks([]), plt.yticks([])
    plt.subplot(324), plt.imshow(out),plt.title('Salt & Pepper Noise')
    plt.xticks([]), plt.yticks([])
    plt.subplot(325), plt.imshow(poisson),plt.title('Poisson Noise')
    plt.xticks([]), plt.yticks([])
    plt.subplot(326), plt.imshow(speckle),plt.title('Speckle Noise')
    plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    if settings.seed is None:
        seed = "None"
    else:
        rn.seed(settings.seed)
        seed = str(settings.seed)

    varCode128 = barcode.get_barcode_class('code128')
    # create output directory if needed
    if not os.path.exists("./output"):
        os.makedirs("./output")
    # create textfile to put values of the codes in it
    textfile = open("./output/values.txt", "w+")
    # write information of dataset
    textfile.write("===This dataset contains %i images and was created with random seed: [%s] on the %s===\n"
                   % (settings.dataset_size, seed, str(datetime.datetime.now())[:19]))
    textfile.write("===For more information go to: https://github.com/OnlyRightNow/barcode-synthesizer===\n")
    for i in range(0, settings.dataset_size):

        # create random value with n digits
        value = str(random_with_n_digits(settings.n_digits))
        if settings.show_text:
            text = value
        else:
            # TODO: make sure no text is printed below the barcode (doesn't work yet)
            text = " "
        code128 = varCode128(value, writer=ImageWriter())
        filename = "./output/%s_%s" % (i, value)
        code128.save(filename)
        textfile.write("%s \n" % value)
        img = cv2.imread("%s.png" % filename)
        #out = img[10:190, 10:230]#20:150
        out = img[10:190, 10:240]
        # blur e.g. from printing
        # blur = cv2.blur(img, (2, 2))
        # # salt and pepper noise
        # row, col, ch = blur.shape
        # s_vs_p = 0.5
        # amount = 0.004
        # out = np.copy(blur)
        # # Salt mode
        # num_salt = np.ceil(amount * img.size * s_vs_p)
        # coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
        # out[coords] = 1
        # # Pepper mode
        # num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
        # coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
        # out[coords] = 0
        # # rotate image
        # out = rotate3DImage(out, rn.uniform(-10, 10), rn.uniform(-10, 10),
        #                     rn.uniform(-5, 5), 0, 0, 200, 200)
        # # contrast and brightness
        # out = change_contrast_brightness(out, rn.uniform(0.7, 2), rn.uniform(-127, 127))
        # # motion blur
        # out = cv2.blur(out, (rn.randrange(1, 3), rn.randrange(1, 10)))
        img2 = cv2.resize(out, settings.image_size)
        cv2.imwrite("%s.png" % filename, out)
    # write information of dataset
    textfile.write("===This dataset contains %i images and was created with random seed: [%s] on the %s===\n"
                   % (settings.dataset_size, seed, str(datetime.datetime.now())[:19]))
    textfile.write("===For more information go to: https://github.com/OnlyRightNow/barcode-synthesizer===\n")
    # close textfile
    textfile.close()
