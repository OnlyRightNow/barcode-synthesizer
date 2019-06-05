import barcode
from barcode.writer import ImageWriter
import numpy as np
import cv2
from matplotlib import pyplot as plt


def rotate_image(img, angle):
    """
    :param img: image to rotate
    :param angle: rotating angle
    :return: rotated image with white background
    """
    rows, cols, ch = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    dst = cv2.warpAffine(out, M, (cols, rows), borderValue=(255, 255, 255))
    return dst


def rotate3DImage(input, alpha, beta, gamma, dx, dy, dz, f):
    """
    coordinate system:
    y
    ^
    |
    |
    |z(pointing inwards)--------->x

    :param input: image
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
    # create barcodes
    varCode128 = barcode.get_barcode_class('code128')
    code128 = varCode128("example", writer=ImageWriter())
    code128.save("./output/example")
    for i in range(0, 5):
        value = "%i" % i
        code128 = varCode128(value, writer=ImageWriter())
        filename = "./output/%s" % value
        code128.save(filename)

    img = cv2.imread('./output/example.png')
    # blur
    blur = cv2.blur(img, (2, 2))
    # salt and pepper noise
    row, col, ch = blur.shape
    s_vs_p = 0.5
    amount = 0.004
    out = np.copy(blur)
    # Salt mode
    num_salt = np.ceil(amount * img.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    out[coords] = 1
    # Pepper mode
    num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    out[coords] = 0

    #out = rotate_image(out, -5)
    out = rotate3DImage(out, 10, 10, 5, 0, 0, 200, 200)
    # contrast

    # brightness
    cv2.imwrite('./output/example.png', out)
