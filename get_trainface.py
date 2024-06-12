import cv2
#打开笔记本的内置摄像头
cap = cv2.VideoCapture(0)
i = 0
#也可写成while True
while(1):
    """
    ret：True或者False，代表有没有读取到图片
    frame：表示截取到一帧的图片
    """
    ret,frame = cap.read()
    # 展示图片
    cv2.imshow('capture',frame)
    # 保存图片
    cv2.imwrite(r"D:\trainfece\foure\\"+ str(i) + ".jpg",frame)
    i = i + 1
    """
       cv2.waitKey(1)：waitKey()函数功能是不断刷新图像，返回值为当前键盘的值
       OxFF：是一个位掩码，一旦使用了掩码，就可以检查它是否是相应的值
       ord('q')：返回q对应的unicode码对应的值(113)
    """
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#释放对象和销毁窗口
cap.release()
cv2.destroyAllWindows()
