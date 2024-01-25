import tkinter as tk
from tkinter import *
from tkinter import filedialog
import tkinter.font as font
import tensorflow as tf
import os
import numpy as np
from skimage import io
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from PIL import ImageTk, Image
from numpy import asarray


win = Tk()
myFont = font.Font(family='Bauhaus 93', size=18)
myFont2 = font.Font(family='Bauhaus 93', size=15)
load_model = tf.keras.models.load_model(".\\model.h5", compile=False)
win.title('Сегмнетирование почки')
win.geometry("900x600")
win['bg'] = 'black'

file = None

#Модуль сегментации
def unet(filen):
    global t3
    seed = 42
    np.random.seed = seed
    print(filen)
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    IMG_CHANNELS = 1
    image = imread(filen)[:,:]
    image = resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    image = image[np.newaxis, :, :]
    X_test = asarray(image)
    print(X_test[0].shape)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    preds_test_load = load_model.predict(X_test, verbose=1)
    preds_test_load_t = (preds_test_load > 0.5).astype(np.bool)
    print(preds_test_load_t.shape)
    plt.imshow(np.squeeze(preds_test_load_t[0]))
    plt.savefig("filenames.png")
    l2 = Image.open("filenames.png")
    img = ImageTk.PhotoImage(l2)
    label_1 = Label(win, text='Сгенерированная маска', bg='black', fg='white')
    label_1['font'] = myFont2
    label_1.place(x=440, y=90)
    label = Label(image=img, height=400, width=400)
    label.image = img
    label.place(x=440, y=120)

#Модуль загрузки
def upload():
    global filen
    path = filedialog.askopenfilename(title="выберете КТ-изображение")
    filen = path
    label_1 = Label(win, text='Исходное изображение', bg='black', fg='white')
    label_1['font'] = myFont2
    label_1.place(x=30, y=90)

    image = Image.open(filen)
    resize_image = image.resize((400, 400), Image.ANTIALIAS)
    image = ImageTk.PhotoImage(resize_image)
    label = Label(image=image, height=400, width=400)
    label.image = image
    label.place(x=30, y=120)

def save():
    print("save")
    l3 = Image.open("filenames.png")
    ext = tk.StringVar()
    filename = filedialog.asksaveasfilename(initialdir="/",  title="выберете путь для сохранения", typevariable=ext, filetypes=(('PNG', '.png'), ("all files", "*.*")))
    print(filename)
    print(os.path.dirname(filename))
    print(os.path.basename(filename))
    l3.save(os.path.dirname(filename)+"/"+os.path.basename(filename)+"."+ext.get().lower())

btn_1 = Button(win, text='Загрузить', bg='white', fg='black', height=1, width=10, command=upload)
btn_1['font'] = myFont
btn_1.place(x=30, y=30)

btn_2 = Button(win, text='Сегментировать', bg='white', fg='black', height=1, width=15, command=lambda: unet(filen))
btn_2['font'] = myFont
btn_2.place(x=180, y=30)

btn_3 = Button(win, text='Скачать', bg='white', fg='black', height=1, width=15, command=lambda: save())
btn_3['font'] = myFont
btn_3.place(x=400, y=30)

win.mainloop()