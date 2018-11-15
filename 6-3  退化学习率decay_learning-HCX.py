# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 15:52:30 2018

@author: LDog
"""

#2018-11-15 HCX at WHU LIESMARS 220 Computer Room
#学习使用退化学习率

#实例描述：
#定义一个学习率变量，将其衰减系数设置好，并设置好迭代循环的次数，将每次迭代运算的次数与学习率打印出来
#观察学习率按照次数退化的现象

import tensorflow as tf
global_step=tf.Variable(0,trainable=False) #迭代循环计数变量
initial_learning_rate=0.1 #初始学习率
#令初始学习率以每10次衰减0.9的速度来进行退化
learning_rate=tf.train.exponential_decay(initial_learning_rate,global_step=global_step,
                                         decay_steps=10,decay_rate=0.9)
            
#一般常用的梯度下降算法，一种常用的训练策略，在训练神经网络时，通常在训练刚开始时，使用较大的learning rate
#随着训练的进行，会慢慢减小learning rate，在使用时，一定要把当前迭代次数global——step传进去，否则不会有退化的功能                            
opt=tf.train.GradientDescentOptimizer(learning_rate)

#定义一个op，令global——step加1完成计步
add_global=global_step.assign_add(1)


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(learning_rate))
    for i in range(20):   #可以尝试不同的值
        #循环20步，将每步的学习率打印出来
        g,rate=sess.run([add_global,learning_rate]) 
        print(g,rate) #第11次变成第10次的0.9倍大小


#对梯度下降算法的总结
#梯度下降算法是一个最优化的算法，通常称为最速下降法，常用于机器学习和人工智能中递归性地逼近最小偏差模型
#梯度下降算法的方向是用“负梯度方向”为“搜索方向”，沿着梯度下降的方向求解极小值。
#在训练中，每次的正向传播后都会得到输出值与真实值的损失值，这个损失值越小，代表模型越好
#于是，梯度下降的算法就用在这里，帮助寻找最小的那个损失，从而反推出对应的学习参数b和w，达到优化模型的效果
#常用的梯度下降算法可以分为如下：、
#1）批量梯度下降：遍历全部数据集算一次损失函数，然后算函数对各个参数的梯度和跟新梯度，这种方法
#没更新一次参数，都要把数据集里的所有样本遍历看一遍，计算量大，计算速度慢，不支持在线学习，
#称为Batch Gradient Descent,批梯度下降。
#2）随机梯度下降：每看一个数据就算一下损失函数，然后求梯度更新算法，被称为：
#Stochastic Gradient Descent,这个方法速度比较快，但是收敛性能不太好，可能在最优点附近
#慌来慌去，命中不到最优点。两次参数的跟新也有可能互相抵消，造成目标函数震荡比较剧烈
#3）小批量梯度下降：为了克服上面两种方法的缺点，一般采用一种折中手段--小批的梯度下降。
#这种方法把数据分为若干批，按批次来更新参数，这样一批中的一组数据共同决定了本次梯度的方向，
#下降起来就不容易跑偏，减少了随机性。另一方面因为批的样本数与整个数据集相比小了很多，计算量也不是很大。
                                






