# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 21:29:01 2018

@author: LDog
"""
#2018-11-08 HCX at WHU LIESMARS 220 Computer Room
#检查点：在训练中间的参数需要保存下来的模型

#TensorFlow提供了一个####TensorBoard######，可以将训练过程中的各种数据展示出来，如：
#标量Scalars，图片Images，音频Audio,计算图Graph，数据分布、直方图Histograms和嵌入式向量

#TensorBoard是一个日志展示系统，需要在session中运算图时，将各种类型的数据汇总并输出到日志文件中，
#然后启动TensorBoard服务，TensorBoard读取这些日志文件，并开启6006端口提供Web服务，让用户可以在
#浏览器中查看数据


#通过添加一个标量和一个直方图数据到log里，然后通过TensorBoard显示出来

import tensorflow as tf
import numpy as np
#import random
import matplotlib.pyplot as plt
import datetime  #计时


start1=datetime.datetime.now()#开始计算程序所需总时间

#----------定义生成loss可视化的函数#-----------
plotdata={'batchsize':[],'loss':[]}
def moving_average(a,w=10):
    if len(a) < w: #判断长度
        return a[:]
    return [val if idx< w else sum(a[(idx-w):idx])/w for idx,val in enumerate(a)]
#enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中,例如：
#seasons = ['Spring', 'Summer', 'Fall', 'Winter']
#list(enumerate(seasons start=1))       # 下标从 1 开始
#[(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
#-----------定义生成loss可视化的函数-----------
    

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

tf.reset_default_graph()# 重置图

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

tf.summary.histogram('z',z) #将预测值以直方图形式显示！！！！！！！

#反向优化
cost=tf.reduce_mean(tf.square(Y-z)) #平方差

tf.summary.scalar('losss_function',cost) #将损失以标量形式显示！！！！！！！

learning_rate=0.01 #学习率
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)#梯度下降

#-----------3-迭代训练---------------------------
#1.训练模型
#初始化所有变量
init=tf.global_variables_initializer()
#定义参数
training_epochs=20   #迭代参数
display_step=2  #打印结果的步长

saver=tf.train.Saver(max_to_keep=1)  #在session中通过saver的save即可将模型保存起来,保存1次
#在迭代过程中只保存一个文件，顾在循环训练过程中，新生成的模型就会覆盖以前的模型
savedir='log/' #保存路径


#启动session
with tf.Session() as sess:
    sess.run(init) #启动初始化

    merged_summary_op=tf.summary.merge_all() #合并所有summary！！！！
    #创建summary_writer,用于写文件
    summary_writer=tf.summary.FileWriter('log/mnist_with_summaries',sess.graph)
    
    plotdata={'batchsize':[],'loss':[]}#存放批次值和损失值
    #向模型输入数据
    print('\n','开始迭代：','\n')
    for epoch in range(training_epochs):
        time_start=datetime.datetime.now() #开始计时
        for (x,y) in zip(train_X,train_Y):   #x,y是每次取出的值，避免用两个循环？
        #zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。            
#            print(zip(x,y))
            sess.run(optimizer,feed_dict={X:x,Y:y}) #运行会话，并传参数
        
        #生成summary
        summary_str=sess.run(merged_summary_op,feed_dict={X:x,Y:y});
        #将summary写入文件
        summary_writer.add_summary(summary_str,epoch)
         
        #显示训练中的详细信息
        if epoch % display_step ==0:
            loss=sess.run(cost,feed_dict={X:train_X,Y:train_Y})
            time_end=datetime.datetime.now()  #结束计时
            time=time_end-time_start

            print('Epoch:',epoch+1,'cost=',loss,'W=',sess.run(W),'b=',sess.run(b),'time=',time)
            if not (loss=='NA'):
                plotdata['batchsize'].append(epoch)
                plotdata['loss'].append(loss)
                 #保存结果 线性模型   需要在循环中实现
                saver.save(sess,savedir+'linermodel.cpkt',global_step=epoch)    

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

#重启一个session ,载入一个检查点 
load_epoch=18
with tf.Session() as sess2:
    sess2.run(tf.global_variables_initializer())
    #tensorflow在载入时，同样也需要制定迭代次数
    saver.restore(sess2,savedir+'linermodel.cpkt-'+str(load_epoch))
    print('x=0.3,z=',sess2.run(z,feed_dict={X:0.2}))
  
end1=datetime.datetime.now() #结束计算程序所需总时间 
print('程序所需总时间为：',str(end1-start1))






#在summary日志的上级路径下，输入如下命令：

#tensorboard --logdir mnist_with_summaries/ Starting TensorBoard b'54' at http://PC:6006

#tensorboard --logdir mnist_with_summaries/ --port 6007


#浏览器最好选用Chrome
#在命令行里启动TensorBoard时，一定要先进入到日志所在的上级路径下，否则打开的页面里找不到创建好的信息



  













