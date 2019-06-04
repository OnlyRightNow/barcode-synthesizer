import barcode
from barcode.writer import ImageWriter
import numpy as np
import cv2
from matplotlib import pyplot as plt

# create barcodes
varCode128 = barcode.get_barcode_class('code128')
value = "example"
code128 = varCode128(value, writer=ImageWriter())
filename = "./output/%s" % value
code128.save(filename)

for i in range(0, 5):
    value = "%i" % i
    code128 = varCode128(value, writer=ImageWriter())
    filename = "./output/%s" % value
    code128.save(filename)


# make them look realistic
img = cv2.imread('./output/0.png')

# blur
blur = cv2.blur(img, (5, 5))

# gaussian noise
row,col,ch= img.shape
mean = 0
var = 0.1
sigma = var**0.5
gauss = np.random.normal(mean,sigma,(row,col,ch))
gauss = gauss.reshape(row,col,ch)
noisy = img + gauss

# salt and pepper noise
row,col,ch = img.shape
s_vs_p = 0.5
amount = 0.004
out = np.copy(img)
# Salt mode
num_salt = np.ceil(amount * img.size * s_vs_p)
coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
out[coords] = 1
# Pepper mode
num_pepper = np.ceil(amount* img.size * (1. - s_vs_p))
coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
out[coords] = 0

# poisson noise
vals = len(np.unique(img))
vals = 2 ** np.ceil(np.log2(vals))
poisson = np.random.poisson(img * vals) / float(vals)

# speckle noise
row,col,ch = img.shape
gauss = np.random.randn(row,col,ch)
gauss = gauss.reshape(row,col,ch)
speckle = img + img * gauss


plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(speckle),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()

