import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from skimage import morphology
import keras
from keras import layers
from keras.models import Model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imgaug import augmenters as iaa
import random
from keras import optimizers


# print(x_real.shape, y_real.shape)
# plt.figure("Image")
# plt.imshow(x_real[0].squeeze(), cmap='gray')
# plt.title('image')
# plt.show()


def calcGrayHist(I):
    h, w = I.shape[:2]
    grayHist = np.zeros([256], np.uint64)
    for i in range(h):
        for j in range(w):
            grayHist[I[i][j]] += 1

    return grayHist


def preprocess(image_input):
    # result = np.zeros(image.shape, dtype=np.float32)
    # cv.normalize(image, result, alpha=0, beta=100, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    guassian = cv.GaussianBlur(image_input, (3, 3), 0)
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    imageEnhance = cv.filter2D(guassian, -1, kernel)
    imageEnhance = cv.filter2D(imageEnhance, -1, kernel)

    # Canny = cv.Canny(image, 100, 180)
    _, img_b = cv.threshold(imageEnhance, 80, 255, cv.THRESH_BINARY)
    # skel, distance = morphology.medial_axis(cv.bitwise_not(img_b), return_distance=True)
    # dist_on_skel = distance * skel
    # dist_on_skel = dist_on_skel.astype(np.uint8) * 255
    binary = cv.bitwise_not(img_b)
    binary[binary == 255] = 1
    skeleton = morphology.skeletonize(binary)
    skeleton = skeleton.astype(np.uint8) * 255
    # plt.figure("Image")
    # plt.imshow(skeleton, cmap='gray')
    # plt.title('skeleton')
    # plt.show()
    return skeleton
    # plt.figure("Image")
    # plt.imshow(image, cmap='gray')
    # plt.title('origin')
    # plt.show()
    # plt.figure("Image")
    # plt.imshow(dist_on_skel, cmap='gray')
    # plt.title('skel')
    # plt.show()

    # plt.figure("Image")
    # plt.imshow(img_b, cmap='gray')
    # plt.title('enhance_b')
    # plt.show()
    # _, img_c = cv.threshold(imageConnect, 170, 255, cv.THRESH_BINARY)
    # grayHist_g = calcGrayHist(guassian)
    # grayHist_o = calcGrayHist(image)
    # x = np.arange(256)
    # plt.plot(x, grayHist_g, 'b', linewidth=2)
    #
    # plt.plot(x, grayHist_o, 'r', linewidth=2)
    # plt.xlabel("gray Label")
    # plt.ylabel("number of pixels")
    # plt.show()
    # cv.imshow("origin", image)
    # cv.imshow("dilate_image", dilate_img)
    # cv.imshow("enhance", img_b)
    # cv.imshow("dilate", img_d)
    # cv.imshow("guassian", guassian)
    # cv.imshow("enhance", imageEnhance)
    # cv.imshow("hance", imageEnhance)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


def transfer(data):
    result = []
    for i in range(data.shape[0]):
        result.append(preprocess(data[i].squeeze()))
    return np.array(result)


x_real = np.load('x_real.npz')['data'][:500:]
y_real = np.load('y_real.npy')[:500:]

grayHist = calcGrayHist(x_real[0].squeeze())

x_train = transfer(x_real)
# print(x_train.shape)
x_range = range(256)
# plt.plot(x_range, grayHist, 'r', linewidth=2)
# plt.figure("Image")
# plt.imshow(x_real[0], cmap='gray')
# plt.title('origin')
# plt.show()
new_x_train = x_train[0] * (1 / 255)


# print(np.shape(new_x_train))


def neighbour(L, x, y):
    nb_neighbour = 0
    if (L[x][y] == 1):
        for i in range(-1, 2):
            for j in range(-1, 2):
                nb_neighbour += L[x + i][y + j]
        return nb_neighbour - L[x][y]
    else:
        return 0


###
# Commentaire
# plt.imshow(x_train[0], cmap='gray')
plt.title('skeleton')


# ça arrête le programme ici
# plt.show()

# plt.figure("Image")
# plt.imshow(x_train[0], cmap='gray')
# plt.title('skeleton')
# plt.show()

# Permet d'obtenir à partir de la matrice d'origine, la matrice CN
def crossing_number(Matrice):
    height, width = np.shape(Matrice)
    L = np.zeros((height, width))
    transition = []
    terminaison = []
    bifurcation = []
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if neighbour(Matrice, i, j) == 1:
                terminaison.append([i, j])
                L[i][j] = 1
            elif neighbour(Matrice, i, j) == 2:
                transition.append([i, j])
                L[i][j] = 0
            elif neighbour(Matrice, i, j) > 2:
                bifurcation.append([i, j])
                L[i][j] = 3
    return L


# Permet d'obtenir les terminaisons
def extration_terminaison(Matrice):
    height, width = np.shape(Matrice)
    L = np.zeros((height, width))
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if (neighbour(Matrice, i, j) == 1):
                L[i][j] = 255
    return L


# Permet d'obtenir les transitions
def extration_transition(Matrice):
    height, width = np.shape(Matrice)
    L = np.zeros((height, width))
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if (neighbour(Matrice, i, j) == 2):
                L[i][j] = 255
    return L


# Permet d'obtenir les bifucations
def extration_bifurcation(Matrice):
    height, width = np.shape(Matrice)
    L = np.zeros((height, width))
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if (neighbour(Matrice, i, j) > 2):
                L[i][j] = 255
    return L


# print(np.shape(new_x_train))
# print(crossing_number(new_x_train))
# plt.imshow(crossing_number(new_x_train))
# plt.title('CN')
# plt.show()
# image = x_real[0].squeeze()
# preprocess(image)
# image = x_real[1].squeeze()
# preprocess(image)
# image = x_real[20].squeeze()
# preprocess(image)
# print(y_real[0])
# print(y_real[1])
# print(y_real[20])

# ====================
# coding = utf-8

# Prepare data X_train: ndarray,(60000, 28, 28) y_train: ndarray, (60000,)
y_input = np.zeros(20)
y_input[0:10] = 0
y_input[10:20] = 1
print("y_input:")
print(y_input[0].shape)
# plt.imshow(x_train[0])
x_train_2 = x_train[:20:] * (1 / 255)

# 用np定义一个90*90的矩阵，用于存放图片

X_input = np.zeros((20, 90, 90))


# print(crossing_number(x_train_2[0]))


def transfer2(data):
    result = []
    for i in range(20):
        result.append(crossing_number(data[i]))
    return np.array(result)


x_input = transfer2(x_train_2)

plt.imshow(x_input[0])
plt.imshow(crossing_number(x_input[0]))
plt.show()
print(x_input.shape)
print(x_input[0][51])
plt.imshow(x_train_2[0])

(x_train, y_train) = (x_input, y_input)
x_train_real, x_valid, y_train_real, y_valid = train_test_split(x_train, y_train, test_size=0.2)

# ====================
# Build model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=[90, 90]),  # input layer
    keras.layers.Dense(100, activation='relu'),  # hidden layer
    keras.layers.Dropout(0.2),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(50, activation='relu'),  # hidden layer
    keras.layers.Dropout(0.2),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(2, activation='softmax')  # output layer
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(x_train_real, y_train_real, epochs=100)

# Evaluate model
valid_loss, valid_acc = model.evaluate(x_valid, y_valid, verbose=1)
print(f"Valid loss:{valid_loss}")
print(f"Valid accuracy:{valid_acc}")

# Make one prediction
class_names = ['First Person', 'Second Person']
y_predicts = model.predict(x_valid)
y_index = np.argmax(y_predicts[0])
y_label = class_names[y_index]
print("Number 1 is: ", y_label)
y_index = np.argmax(y_predicts[1])
y_label = class_names[y_index]
print("Number 2 is: ", y_label)
y_index = np.argmax(y_predicts[2])
y_label = class_names[y_index]
print("Number 3 is: ", y_label)
y_index = np.argmax(y_predicts[3])
y_label = class_names[y_index]
print("Number 4 is: ", y_label)
