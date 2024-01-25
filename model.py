import tensorflow as tf
import os
import random
import numpy as np
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from google.colab import drive
from PIL import Image, ImageEnhance
from skimage import io, transform
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

drive.mount('/content/drive')

# обработка данных
seed = 42
np.random.seed = seed
TRAIN_IMAGE_PATH = '/content/drive/MyDrive/box3/img'
train_ids = next(os.walk(TRAIN_IMAGE_PATH))[2]
train_ids

id_= 0
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
  img = Image.open('/content/drive/MyDrive/box3/img'+ '/' + id_)
  img.save('/content/drive/MyDrive/box_pre/img_pre'+ '/' + id_.split('.')[0]+'.png', 'png')

id_ = 0
TRAIN_IMAGE_PATH = '/content/drive/MyDrive/box_pre/img_pre'
#TRAIN_IMAGE_MASK = '/content/mask_pre'
train_ids = next(os.walk(TRAIN_IMAGE_PATH))[2]
train_ids

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
  img = Image.open('/content/drive/MyDrive/box3/masks_machine'+ '/' + id_)
  factor = 255
  enchancer = ImageEnhance.Brightness(img)
  img2 = enchancer.enhance(factor)
  img2.save('/content/drive/MyDrive/box_pre/mask_pre'+ '/' + id_)

id_ = 0
TRAIN_IMAGE_PATH = '/content/drive/MyDrive/box_pre/img_pre'
TRAIN_IMAGE_MASK = '/content/drive/MyDrive/box_pre/mask_pre'
train_ids = next(os.walk(TRAIN_IMAGE_PATH))[2]

#обучение модели
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
IMG_CHANNELS2 = 1

X = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
Y = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH), dtype=np.bool)
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    img = imread(TRAIN_IMAGE_PATH + '/' + id_)[:,:]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X[n] = img  #Fill empty X_train with values from img

    mask = imread(TRAIN_IMAGE_MASK + '/' + id_)[:,:]
    mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    Y[n] = mask

X_train, X_test, y_train, y_test = train_test_split (X, Y, test_size=0.2, random_state=42)

rand_x = random.randint(0, 374)
print(rand_x)
imshow(X_train[rand_x])
plt.show()
print(X_train[rand_x].shape, '   ', y_train[rand_x].shape,'', rand_x)
imshow(np.squeeze(y_train[rand_x]))
plt.show()
plt.show()

inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS2))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

#Contraction path
c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

c4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

c5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path
u6 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

def dice_bce_loss(y_pred, y_true):
    intersection = tf.reduce_sum(y_true * y_pred)
    dice = (2. * intersection) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))
    total_loss = 0.25 * dice + tf.keras.losses.binary_crossentropy(y_pred, y_true)
    return total_loss

model1 = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model1.compile(optimizer='adam', loss=dice_bce_loss, metrics=['accuracy'])
model1.summary()

X_train2 = X_train.astype(np.float)
y_train2 = y_train.astype(np.float)

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=30, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='drive/MyDrive/logs_tb_pish',   histogram_freq=1, write_images=True)]
results = model1.fit(X_train2, y_train2, validation_split=0.1, batch_size=16, epochs=50, callbacks=callbacks)

%load_ext tensorboard
%tensorboard --logdir 'drive/MyDrive/logs_tb_pish'

preds_val = model1.predict(X_train2[int(X_train2.shape[0]*0.9):], verbose=1)
preds_val_t = (preds_val > 0.5).astype(np.bool)
preds_val_S = model1.predict(X_train2).flatten()
Y_train_f = y_train.flatten()
preds_val_S_t = (preds_val_S > 0.5).astype(np.bool)

report = classification_report(Y_train_f, preds_val_S_t)
print(report, '\n')
DICE = dice_coefficient(Y_train_f, preds_val_S_t)
print("DICE: ", DICE, "%")
jac_index = 0
for i in range(len(preds_val_t)):
    intersect = 0
    union = 0
    x = preds_val_t[i].flatten()
    y = y_train2[int(X_train2.shape[0]*0.9) + i].flatten()
    for j in range(len(x)):
        if x[j] == 1 and y[j] == 1 :
            intersect += 1
            union += 1
        elif (x[j] == 1 and y[j] == 0) or (x[j] == 0 and y[j] == 1):
            union += 1
    jac_index += intersect*100.0/union

print("Jaccard index:", jac_index / len(preds_val_t), " %")
#print('\n', "DICE score"dice_bce_loss(Y_train2, preds_val_t))

ix = random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train2.shape[0]*0.9) + ix])
plt.show()
imshow(np.squeeze(y_train2[int(X_train2.shape[0]*0.9) + ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()

preds_test = model1.predict(X_test, verbose=1)
preds_test_t = (preds_test > 0.5).astype(np.bool)
preds_val_S = model1.predict(X_test).flatten()
Y_test_f = y_test.flatten()
preds_val_S_t = (preds_val_S > 0.5).astype(np.bool)

report = classification_report(Y_test_f, preds_val_S_t)
print(report, '\n')
DICE = dice_coefficient(Y_test_f, preds_val_S_t)
print("DICE: ", DICE, "%")

jac_index = 0
for i in range(len(preds_test_t)):
    intersect = 0
    union = 0
    x = preds_test_t[i].flatten()
    y = y_test[i].flatten()
    for j in range(len(x)):
        if x[j] == 1 and y[j] == 1 :
            intersect += 1
            union += 1
        elif (x[j] == 1 and y[j] == 0) or (x[j] == 0 and y[j] == 1):
            union += 1
    jac_index += intersect*100.0/union

print("Jaccard index:", jac_index / len(preds_test_t), " %")
ix = random.randint(0, len(preds_test_t))
imshow(np.squeeze(X_test[ix]))
plt.show()
imshow(np.squeeze(np.squeeze(y_test[ix])))
plt.show()
imshow(np.squeeze(np.squeeze(preds_test_t[ix])))
plt.show()

