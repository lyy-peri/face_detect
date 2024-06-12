import cv2
import os
import sys
import random

out_dir = './Face_date1/other'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# 改变亮度与对比度
def relight(img, alpha=1, bias=0):
    w = img.shape[1]
    h = img.shape[0]
    #image = []
    for i in range(0,w):
        for j in range(0,h):
            for c in range(3):
                tmp = int(img[j,i,c]*alpha + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j,i,c] = tmp
    return img

# 获取分类器
haar = cv2.CascadeClassifier("C:\\Users\\peri\\Desktop\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml")

# 打开摄像头 参数为输入流，可以为摄像头或视频文件
camera = cv2.VideoCapture(0)

n = 1
while 1:
    if (n <= 100):
        print('It`s processing %s image.' % n)
        # 读帧
        success, img = camera.read()
        #转为灰度图像
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #使用 haar.detectMultiScale 方法在灰度图像 gray_img 中进行人脸检测，并返回检测到的人脸位置信息
        #gray_img：表示输入的灰度图像，通常是将彩色图像转换为灰度图像后进行人脸检测，因为人脸检测通常使用灰度图像来提高检测的准确率和性能
        #1.3：表示每次图像尺度减小的比例。在每次检测过程中，图像会逐渐缩小以寻找不同尺寸的人脸。这个参数可以根据需要进行调整，通常设置在 1.1 到 1.5 之间。
        #5：表示参数minNeighbors，指定在一个目标区域中检测到多少个对象才认为是真正的目标。一般来说，这个值越大，准确率越高，但可能会漏掉一些目标；这个值越小，检测到的目标越多，但也可能会包括一些错误的目标.
        #该方法会返回一个包含检测到的人脸位置信息的列表，每个元素都是一组坐标信息𝑥,𝑦,𝑤,ℎ 。x，y，w，h代表检测到的人脸矩形区域的左上角坐标x，y和宽度w高度h。
        #例如，如果检测到一个人脸，可能会返回类似(x 1，y 1，w 1，h 1)的列表，如果检测到多个人脸，可能会返回类似(x 1，y 1，w 1，h 1），（x 2，y 2，w 2，h 2），...的列表。
        faces = haar.detectMultiScale(gray_img, 1.3, 5)
        #循环处理每一个人脸
        for f_x, f_y, f_w, f_h in faces:
            #根据读取的列表获得每一个人脸的位置，随后只读取人脸范围的位置并重新赋值给face
            face = img[f_y:f_y+f_h, f_x:f_x+f_w]
            #这是图像可视化的大小也就是你获取图像时摄像头展示图片的大小，为什么设置这么小呢？
            #将人脸尺寸设置为64,64的主要原因是为了与神经网络的输入尺寸相匹配。在很多人脸识别和人脸相关任务的深度学习模型中，常常要求输入的图像尺寸是固定的，通常是为了方便模型训练和降低计算复杂度。
            #统一尺寸：保证所有人脸图像的尺寸都是一致的，这样可以简化网络结构，避免在输入时需要进行尺寸变换。
            #加速训练：统一的输入尺寸意味着在训练过程中可以批量处理相同尺寸的数据，从而加快训练速度。
            #降低计算需求：较小的输入尺寸可以减少模型需要处理的数据量，从而降低计算需求，加速模型推理和训练过程。
            face = cv2.resize(face, (64,64))
            '''
            if n % 3 == 1:
                face = relight(face, 1, 50)
            elif n % 3 == 2:
                face = relight(face, 0.5, 0)
            '''
            face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
            cv2.imshow('img', face)
            cv2.imwrite(out_dir+'/'+str(n)+'.jpg', face)
            n+=1
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break
    else:
        break