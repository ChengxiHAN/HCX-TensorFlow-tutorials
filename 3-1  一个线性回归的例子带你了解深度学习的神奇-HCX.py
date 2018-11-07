# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 18:34:26 2018

@author: LDog
"""
#2018-11-07 HCX于 WHU LIESMARS 220 Computer Room

import tensorflow as tf
import numpy as np
#import random
import matplotlib.pyplot as plt

#-----------后边需要用到的一个函数#-----------
plotdata={'batchsize':[],'loss':[]}
def moving_average(a,w=10):
    if len(a) < w: #判断长度
        return a[:]
    return [val if idx< w else sum(a[(idx-w):idx])/w for idx,val in enumerate(a)]
#enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中,例如：
#seasons = ['Spring', 'Summer', 'Fall', 'Winter']
#list(enumerate(seasons start=1))       # 下标从 1 开始
#[(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
#-----------后边需要用到的一个函数#-----------
    

#x=[];n=100;
#while n<=100:
#    x.append(random.randint(-1,1))  # generate random number 产生随机数
#    n+=1
#-----------1-准备数据---------------------------
train_X=np.linspace(-1,1,100) #100为步长
#y=x*2+random.randint(-1,1)*0.3 
train_Y=train_X*2+np.random.randn(*train_X.shape)*0.3  #加入一些扰动

#可视化数据
#plt.plot(train_X,train_Y,'r^',label='Original Data')
#plt.title('Data~Data')
#plt.legend()#显示标签
#plt.rcParams['font.sans-serif']=['SimHei']#用来正常显示中文标签
#plt.rcParams['axes.unicode_minus']=False #用来正常显示负号‘-’
#plt.xlabel('横轴是x')
#plt.ylabel('纵轴是y')
#plt.show
print('已知数据的X是：','\n',train_X,'\n','已知数据的Y是：','\n',train_Y)

#-----------2-搭建模型---------------------------
#创建模型
#占位符
X=tf.placeholder('float')
Y=tf.placeholder('float') #对应的真实Y值
#模型参数
W=tf.Variable(tf.random_normal([1]),name='weight')
b=tf.Variable(tf.zeros([1]),name='bias')
#前向结构
z=tf.multiply(X,W)+b

#反向优化
cost=tf.reduce_mean(tf.square(Y-z)) #平方差
learning_rate=0.01 #学习率
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)#梯度下降

#-----------3-迭代训练---------------------------
#1.训练模型
#初始化所有变量
init=tf.global_variables_initializer()
#定义参数
training_epochs=20   #迭代参数
display_step=2  #打印结果的步长

#启动session
with tf.Session() as sess:
    sess.run(init) #启动初始化
    plotdata={'batchsize':[],'loss':[]}#存放批次值和损失值
    #向模型输入数据
    for epoch in range(training_epochs):
        for (x,y) in zip(train_X,train_Y):   #x,y是每次取出的值，避免用两个循环？
        #zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。            
#            print(zip(x,y))
            sess.run(optimizer,feed_dict={X:x,Y:y}) #运行会话，并传参数
            
        
        #显示训练中的详细信息
        if epoch % display_step ==0:
            loss=sess.run(cost,feed_dict={X:train_X,Y:train_Y})
            print('Epoch:',epoch+1,'cost=',loss,'W=',sess.run(W),'b=',sess.run(b))
            if not (loss=='NA'):
                plotdata['batchsize'].append(epoch)
                plotdata['loss'].append(loss)
                
    print('Finished')
    print('cost=',sess.run(cost,feed_dict={X:train_X,Y:train_Y}),'W=',sess.run(W),'b=',sess.run(b))
    
            
#2.训练模型中的可视化
    plt.subplot(211)
    plt.plot(train_X,train_Y,'r^',label='Original Data')
    plt.plot(train_X,sess.run(W)*train_X+sess.run(b),label='Fittedline()') #拟合线
    plt.legend()
    plt.title('Data~Data')
    plt.rcParams['font.sans-serif']=['SimHei']#用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号‘-’
    plt.xlabel('横轴是x')
    plt.ylabel('纵轴是y')    
    plt.show()
    
    plotdata['avgloss']=moving_average(plotdata['loss'])
    plt.figure(1)
    plt.subplot(212)
    plt.plot(plotdata['batchsize'],plotdata['avgloss'],'g--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss')
    plt.show()
      
#-----------4-使用模型---------------------------
    print('x若=0.2,  z则预测出=',sess.run(z,feed_dict={X:0.2}))
    














