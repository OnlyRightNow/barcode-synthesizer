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

def tilt_image(img, alpha, beta):
    # https://stackoverflow.com/questions/33497736/opencv-adjusting-photo-with-skew-angle-tilt
    # https://stackoverflow.com/questions/17087446/how-to-calculate-perspective-transform-for-opencv-from-rotation-angles

    rows, cols, ch = img.shape
    pts1 = np.float32(
        [[cols * .25, rows * .95],
         [cols * .90, rows * .95],
         [cols * .10, 0],
         [cols, 0]]
    )
    pts2 = np.float32(
        [[cols * 0.1, rows],
         [cols, rows],
         [0, 0],
         [cols, 0]]
    )
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (cols, rows))
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

    out = rotate_image(out, -5)

    out = tilt_image(out, 1, 1)

    cv2.imwrite('./output/example.png', out)
