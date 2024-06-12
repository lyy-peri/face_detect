import os
import random

import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D,Conv2D, MaxPool2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
from sklearn.preprocessing import LabelEncoder

from load_image import load_dataset, resize_image, IMAGE_SIZE

class Dataset:
    def __init__(self, path_name):
        # 训练集
        self.train_images = None
        self.train_labels = None

        # 验证集
        self.valid_images = None
        self.valid_labels = None

        # 测试集
        self.test_images = None
        self.test_labels = None

        # 数据集加载路径
        self.path_name = path_name

        # 当前库采用的维度顺序
        self.input_shape = None

    # 加载数据集并按照交叉验证的原则划分数据集并进行相关预处理工作
    def load(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE,
             img_channels=3, nb_classes=2):
        # 加载数据集到内存
        images, labels = load_dataset(self.path_name)

        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(labels)
        num_classes = len(label_encoder.classes_)
        one_hot_labels = np_utils.to_categorical(integer_encoded, num_classes)

        train_images, valid_images, train_labels, valid_labels = train_test_split(images, one_hot_labels,
                                                                                  train_size=0.5,
                                                                                  random_state=random.randint(0, 100))
        _, test_images, _, test_labels = train_test_split(images, one_hot_labels, test_size=0.5,
                                                          random_state=random.randint(0, 100))

        # 当前的维度顺序如果为'th'，则输入图片数据时的顺序为：channels,rows,cols，否则:rows,cols,channels
        # 这部分代码就是根据keras库要求的维度顺序重组训练数据集
        if K.image_dim_ordering() == 'th':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)

            # 输出训练集、验证集、测试集的数量
            print(train_images.shape[0], 'train samples')
            print(valid_images.shape[0], 'valid samples')
            print(test_images.shape[0], 'test samples')

            # 我们的模型使用categorical_crossentropy作为损失函数，因此需要根据类别数量nb_classes将
            # 类别标签进行one-hot编码使其向量化，在这里我们的类别只有两种，经过转化后标签数据变为二维
            #train_labels = np_utils.to_categorical(train_labels, nb_classes)
            #valid_labels = np_utils.to_categorical(valid_labels, nb_classes)
            #test_labels = np_utils.to_categorical(test_labels, nb_classes)

            # 像素数据浮点化以便归一化
            train_images = train_images.astype('float32')
            valid_images = valid_images.astype('float32')
            test_images = test_images.astype('float32')

            # 将其归一化,图像的各像素值归一化到0~1区间
            train_images /= 255
            valid_images /= 255
            test_images /= 255

            self.train_images = train_images
            self.valid_images = valid_images
            self.test_images = test_images
            self.train_labels = train_labels
            self.valid_labels = valid_labels
            self.test_labels = test_labels


# CNN网络模型类
class Model:
    def __init__(self):
        self.model = None
        # 建立模型
    def build_model(self, dataset, nb_classes):
        # 构建一个空的网络模型，它是一个线性堆叠模型，各神经网络层会被顺序添加，专业名称为序贯模型或线性堆叠模型
        self.model = Sequential(name='vgg16')

        # 第1个卷积区块(block1)
        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=dataset.input_shape, name='block1_conv1'))
        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu', name='block1_conv2'))
        self.model.add(MaxPool2D((2, 2), strides=(2, 2), name='block1_pool'))

        # 第2个卷积区块(block2)
        self.model.add(Conv2D(128, (3, 3), padding='same', activation='relu', name='block2_conv1'))
        self.model.add(Conv2D(128, (3, 3), padding='same', activation='relu', name='block2_conv2'))
        self.model.add(MaxPool2D((2, 2), strides=(2, 2), name='block2_pool'))

        # 第3个区块(block3)
        self.model.add(Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv1'))
        self.model.add(Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv2'))
        self.model.add(Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv3'))
        self.model.add(MaxPool2D((2, 2), strides=(2, 2), name='block3_pool'))

        # 第4个区块(block4)
        self.model.add(Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv1'))
        self.model.add(Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv2'))
        self.model.add(Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv3'))
        self.model.add(MaxPool2D((2, 2), strides=(2, 2), name='block4_pool'))

        # 第5个区块(block5)
        self.model.add(Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv1'))
        self.model.add(Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv2'))
        self.model.add(Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv3'))
        self.model.add(MaxPool2D((2, 2), strides=(2, 2), name='block5_pool'))

        # 前馈全连接区块
        self.model.add(Flatten(name='flatten'))
        self.model.add(Dense(2048, activation='relu', name='fc1'))
        self.model.add(Dense(256, activation='relu', name='fc2'))
        self.model.add(Dense(nb_classes, activation='softmax', name='predictions'))

        # 输出模型概况
        self.model.summary()

    # 训练模型
    def train(self, dataset, batch_size=20, nb_epoch=5, data_augmentation=True):
        sgd = SGD(lr=0.01, decay=1e-6,
                  momentum=0.9, nesterov=True)  # 采用SGD+momentum的优化器进行训练，首先生成一个优化器对象
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])  # 完成实际的模型配置工作

        # 不使用数据提升，所谓的提升就是从我们提供的训练数据中利用旋转、翻转、加噪声等方法创造新的
        # 训练数据，有意识的提升训练数据规模，增加模型训练量
        if not data_augmentation:
            self.model.fit(dataset.train_images,
                           dataset.train_labels,
                           batch_size=batch_size,
                           nb_epoch=nb_epoch,
                           validation_data=(dataset.valid_images, dataset.valid_labels),
                           shuffle=True)
        # 使用实时数据提升
        else:
            # 定义数据生成器用于数据提升，其返回一个生成器对象datagen，datagen每被调用一
            # 次其生成一组数据（顺序生成），节省内存，其实就是python的数据生成器
            datagen = ImageDataGenerator(
                featurewise_center=False,  # 是否使输入数据去中心化（均值为0），
                samplewise_center=False,  # 是否使输入数据的每个样本均值为0
                featurewise_std_normalization=False,  # 是否数据标准化（输入数据除以数据集的标准差）
                samplewise_std_normalization=False,  # 是否将每个样本数据除以自身的标准差
                zca_whitening=False,  # 是否对输入数据施以ZCA白化
                rotation_range=20,  # 数据提升时图片随机转动的角度(范围为0～180)
                width_shift_range=0.2,  # 数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
                height_shift_range=0.2,  # 同上，只不过这里是垂直
                horizontal_flip=True,  # 是否进行随机水平翻转
                vertical_flip=False)  # 是否进行随机垂直翻转

            # 计算整个训练样本集的数量以用于特征值归一化、ZCA白化等处理
            datagen.fit(dataset.train_images)

            # 利用生成器开始训练模型
            self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels,
                                                  batch_size=batch_size),
                                     samples_per_epoch=dataset.train_images.shape[0],
                                     nb_epoch=nb_epoch,
                                     validation_data=(dataset.valid_images, dataset.valid_labels))

    def save_model(self, file_path):
        self.model.save(file_path)

    def load_model(self, file_path):
        self.model = load_model(file_path)

    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose=1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

    # 识别人脸
    def face_predict(self, image):
        # 依然是根据后端系统确定维度顺序
        if K.image_dim_ordering() == 'th' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_image(image)  # 尺寸必须与训练集一致都应该是IMAGE_SIZE x IMAGE_SIZE
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))  # 与模型训练不同，这次只是针对1张图片进行预测
        elif K.image_dim_ordering() == 'tf' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_image(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))

            # 浮点并归一化
        image = image.astype('float32')
        image /= 255

        # 预测并输出多个类别的概率
        result = self.model.predict_proba(image)
        print('result:', result)

        # 给出所有类别的预测结果
        predicted_classes = self.model.predict(image)
        print('predicted_classes:', predicted_classes)

        # 给出最可能的类别
        max_prob_index = np.argmax(predicted_classes)
        max_prob_class = max_prob_index  # 例子，实际应该根据你的类别对应关系来转换
        print('max_prob_class:', max_prob_class)

        # 返回类别预测结果
        return max_prob_class


if __name__ == '__main__':
    dataset = Dataset('./Face_date1')
    dataset.load()
    model = Model()
    model.build_model(dataset,len([name for name in os.listdir('./Face_date1') if os.path.isdir(os.path.join('./Face_date1', name))]))
    model.train(dataset)
    model.save_model(file_path='./model/LYY_face_model_VGG.h5')
    model.evaluate(dataset)
