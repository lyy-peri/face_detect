from tkinter import font
import cv2
import sys
from tkinter import *
import tkinter as tk
master = Tk()
def First_Scriptcallback():
    exec(open(rb'./Dlib_catch.py',encoding='UTF-8').read())
def second_Scriptcallback():
    exec(open(rb'./face_model_train.py', encoding='UTF-8').read())
def third_Scriptcallback():
    exec(open(rb'./Face_recognition.py', encoding='UTF-8').read())

master.title("人脸识别系统")
canvas = tk.Canvas(master, height=200, width = 400)
canvas.pack()

# 创建一个自定义字体
custom_font = font.Font(family="Helvetica", size=12, weight="bold")

firstButton = Button(master, text="采集人脸", command=First_Scriptcallback,font=custom_font,width=30, height=5)
firstButton.pack()

secondButton = Button(master, text="训练人脸模板", command=second_Scriptcallback,font=custom_font,width=30, height=5)
secondButton.pack()

thirdButton = Button(master, text="人脸识别", command=third_Scriptcallback,font=custom_font,width=30, height=5)
thirdButton.pack()


mainloop()