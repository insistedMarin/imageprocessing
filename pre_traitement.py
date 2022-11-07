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


# Calcule de l'Histogramme en niveau de gris
def calcGrayHist(I):
    h, w = I.shape[:2]
    grayHist = np.zeros([256], np.uint64)
    for i in range(h):
        for j in range(w):
            grayHist[I[i][j]] += 1

    return grayHist

# Pré-traitement de l'image
def preprocess(image_input):
    # gussianblur kernel
    guassian = cv.GaussianBlur(image_input, (3, 3), 0)
    # laplace kernel
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    imageEnhance = cv.filter2D(guassian, -1, kernel)
    imageEnhance = cv.filter2D(imageEnhance, -1, kernel)

   # binary 0 255
    _, img_b = cv.threshold(imageEnhance, 80, 255, cv.THRESH_BINARY)
    binary = cv.bitwise_not(img_b)
    binary[binary == 255] = 1
    # binary 0 1
    skeleton = morphology.skeletonize(binary)
    skeleton = skeleton.astype(np.uint8) * 255
    

    return skeleton



# Permettre de transférer les images
def transfer(data):
    result = []
    for i in range(data.shape[0]):
        result.append(preprocess(data[i].squeeze()))
    return np.array(result)


x_real = np.load('x_real.npz')['data'][:500:]
y_real = np.load('y_real.npy')[:500:]

grayHist = calcGrayHist(x_real[0].squeeze())

x_train = transfer(x_real)

x_range = range(256)

new_x_train = x_train[0] * (1 / 255)


# print(np.shape(new_x_train))

# Permettre de calculer les Crossing Number d'un pixel situé en (x,y) dans la matrice L
def calculCN(L,x,y):
    nb_CN = 0
    Liste= [L[x-1][y-1],L[x][y-1],L[x+1][y-1],L[x+1][y],L[x+1][y+1],L[x][y+1],L[x-1][y+1],L[x-1][y]]
    for i in range (len(Liste)):
        nb_CN += abs(Liste[i]-Liste[i-1])
    return (1/2)*nb_CN


###
# Commentaire
# plt.imshow(x_train[0], cmap='gray')
#plt.title('skeleton')


# plt.show()

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
            if calculCN(Matrice, i, j) == 1:
                terminaison.append([i, j])
                L[i][j] = 1
            elif calculCN(Matrice, i, j) == 2:
                transition.append([i, j])
                L[i][j] = 0
            elif calculCN(Matrice, i, j) > 2:
                bifurcation.append([i, j])
                L[i][j] = 3
    return L


# Permet d'obtenir les terminaisons
def extration_terminaison(Matrice):
    height, width = np.shape(Matrice)
    L = np.zeros((height, width))
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if (calculCN(Matrice, i, j) == 1):
                L[i][j] = 255
    return L


# Permet d'obtenir les transitions
def extration_transition(Matrice):
    height, width = np.shape(Matrice)
    L = np.zeros((height, width))
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if (calculCN(Matrice, i, j) == 2):
                L[i][j] = 255
    return L


# Permet d'obtenir les bifucations
def extration_bifurcation(Matrice):
    height, width = np.shape(Matrice)
    L = np.zeros((height, width))
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if (calculCN(Matrice, i, j) > 2):
                L[i][j] = 255
    return L



# plt.imshow(crossing_number(new_x_train))
# plt.title('CN')
# plt.show()


# Prepare data X_train: ndarray,(60000, 28, 28) y_train: ndarray, (60000,)
y_input = np.zeros(20)
y_input[0:10] = 0
y_input[10:20] = 1
print("y_input:")
print(y_input[0].shape)
# plt.imshow(x_train[0])
x_train_2 = x_train[:20:] * (1 / 255)
print(x_train_2.shape)
print(x_train_2[0])



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
# x_train_real, x_valid, y_train_real, y_valid = train_test_split(x_train, y_train, test_size=0.2)

# ====================
# Build model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=[90, 90]),  # input layer
    keras.layers.Dense(100, activation='relu'),  # hidden layer
    keras.layers.Dense(2, activation='softmax')  # output layer
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=100)

# Evaluate model
valid_loss, valid_acc = model.evaluate(x_train, y_train, verbose=1)
print(f"Valid loss:{valid_loss}")
print(f"Valid accuracy:{valid_acc}")

# Make one prediction
class_names = ['First Person', 'Second Person']
y_predicts = model.predict(x_train)
y_index = np.argmax(y_predicts[0])
y_label = class_names[y_index]
print("Number 1 is: ", y_label)
y_index = np.argmax(y_predicts[4])
y_label = class_names[y_index]
print("Number 5 is: ", y_label)
y_index = np.argmax(y_predicts[10])
y_label = class_names[y_index]
print("Number 11 is: ", y_label)
y_index = np.argmax(y_predicts[19])
y_label = class_names[y_index]
print("Number 20 is: ", y_label)
