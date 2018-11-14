# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 21:04:29 2018

@author: LDog
"""
#2018-11-13 HCX at WHU LIESMARS 220 Computer Room
#mnist数据集下载并可视化-HCX.py


#MNIST是一个入门级的计算机视觉数据集，当我们编程时，第一件事往往会想到学习打印HelloWord
#在机器学习入门的领域里，我们会用MNIST数据集来实验各种模型
#MNIST的官网：http://yann.lecun.com/exdb/mnist/         它包含了四个部分:
#
#Training set images: train-images-idx3-ubyte.gz (9.9 MB, 解压后 47 MB, 包含 60,000 个样本)
#Training set labels: train-labels-idx1-ubyte.gz (29 KB, 解压后 60 KB, 包含 60,000 个标签)
#Test set images: t10k-images-idx3-ubyte.gz (1.6 MB, 解压后 7.8 MB, 包含 10,000 个样本)
#Test set labels: t10k-labels-idx1-ubyte.gz (5KB, 解压后 10 KB, 包含 10,000 个标签)
#
#MNIST 数据集来自美国国家标准与技术研究所, National Institute of Standards and Technology (NIST).
# 训练集 (training set) 由来自 250 个不同人手写的数字构成, 其中 50% 是高中学生,
# 50% 来自人口普查局 (the Census Bureau) 的工作人员. 测试集(test set) 也是同样比例的手写数字数据


#TensorFlow提供了一个库，可以直接自动下载与安装MNIST:

from tensorflow.examples.tutorials.mnist import input_data
#自动下载到该路径中
mnist =input_data.read_data_sets('MNIST_data/',one_hot=True) 
#one_hot是独热的意思 ，将样本标签转化为one_hot编码

#MNIST数据集中的图片是28*28pixel，每一幅图片就是1行784（28*28）列的数据
#黑白图片中，黑色的地方的数值为0， 有图案的地方，数值为0~255之间的数字，代表其颜色的深度

print('输入数据：',mnist.train.images)
print('输入数据打shape：',mnist.train.images.shape)  # (55000, 784)

import pylab

#train.images是一个形状为【55000，784】的张量，第一个维度数字用来索引图片，第二个维度数字来索引每张图片
#中的像素点，此张量里的每一个元素，都表示某张图片里的某个像素的强度值，值介于0~255之间
im=mnist.train.images[2]  #读取MNIST数据中具体的某个数据，可以尝试其他值
im=im.reshape(-1,28)
pylab.imshow(im)  #将数据进行可视化
pylab.show()

#MNIST中有三个数据集：mnist.train; mnist.test;mnist.validation
print('输入数据打shape：',mnist.test.images.shape) #(10000, 784)
print('输入数据打shape：',mnist.validation.images.shape)  #(5000, 784)


#训练集：用于训练
#测试集：用于评估训练过程中的准确度
#验证集：用于评估最终模型的准确度










