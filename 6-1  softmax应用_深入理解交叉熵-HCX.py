# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 21:05:18 2018

@author: LDog
"""

#2018-11-14 HCX at WHU LIESMARS 220 Computer Room

#引言
#交叉熵在TensorFlow中会被封装成多个版本，有的公式里直接带了交叉熵，有的需要自己单独求出
#而在构建模型的时候，如果对没有搞懂这里的差别，出现问题时会很难分析是模型的问题还算交叉熵的问题

#实验描述：
#假设有一个标签labels和一个网络输出值logits
#1）两次softmax实验，将输出值logits分别进行1次和2次softmax，观察两次的区别及意义
#2）观察交叉熵：将步骤1）中的两个值分别进行soft_cross_entropy_with_logits,观察区别
#3）自建公式实验：将做两次softmax的值放到自建组合的公式里得到正确的值

import tensorflow as tf

#已知数据
labels=[[0, 0, 1],[ 0, 1, 0]]
logits=[[2,0.5,6],[0.1,0, 3]]

logits_scaled=tf.nn.softmax(logits)
#进行第二次softmax
logits_scaled2=tf.nn.softmax(logits_scaled)

result1=tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)
result2=tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits_scaled)
result3=-tf.reduce_sum(labels*tf.log(logits_scaled),1) 
#tf.reduce_sum计算输入的tensor元素的和，或者按照axis指定的轴进行求和 0为把列求和； 1为把行求和

with tf.Session() as sess:
    print('第一次计算softmax：scaled=','\n',sess.run(logits_scaled))
    #进过第二次softmax后，分布概率会有变化
    print('进过第二次softmax后，分布概率会有变化：scaled2=','\n',sess.run(logits_scaled2),'\n')
    
    print('正确的方式的结果:rel1=',sess.run(result1),'\n') #正确的方式
    
    #如果将softmax变换完的值放进去会相当于第二次计算softmax的loss，会出错
    print('多计算一次的结果：rel2=',sess.run(result2),'\n')
    
    print('自己构建正确的结果：rel3=',sess.run(result3),'\n')


#小TIP：
#从结果可以看出来，logits里面的值原本加和都是大于1的，但是经过softmax之后，总和变成了1
#样本中的第一个是跟标签分类相符的，第二与标签分类不符，所以第一个的交叉熵比较小为0.02
#第二个比较大为3.09
#
#总结:

#比较scaled和scaled2可以看出来：经过第二次的softmax后，分布概率会有变化，而scaled才是我们真实转化的softmax值
#比较rel1和rel2可以看出：传入softmax_cross_entropy_with_logits的logits是不需要进行softmax，如果将softmax
#后的值scaled传入 softmax_cross_entropy_with_logits就相当于进行了两次softmax转换  



#对非one-hot编码为标签的数据进行交叉熵的计算，比较其与onehot编码的交叉熵之间的差别
labels2=[[0.4,0.1,0.5],[0.3,0.6,0.1]]
result4=tf.nn.softmax_cross_entropy_with_logits(labels=labels2,logits=logits)
with tf.Session() as sess:
    print('标准onehot编码交叉熵的结果：rel4=',sess.run(result4),'\n')

#结果发现：比较前面的rel1发现，对于正确分类的交叉熵和错误分类的交叉熵，二者的结果差别没有标准的onehot那么明显



#了解sparse交叉熵的使用
#sparse_softmax_cross_entropy_with_logits函数的用法不需要使用“非one-hot”的标签
#把标签换成具体的数字【2，1】，对比其与one-hot标注在使用上的区别

#sparse标签
labels3=[2,1] #表明labels中总共分为3个类：0、1、2 。【2，1】等价于one-hot的001和010
result5=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels3,logits=logits)
with tf.Session() as sess:
    print('sparse交叉熵的结果:rel5=',sess.run(result5),'\n') 
    
#总结： 发现result5与rel1的结果完全一样


#计算loss值
#在真正的神经网络中，得到代码中的一个数组并不能满足要求，还需要对其进行“求平均值”，使其最终变成一个具体的数值

#小实验描述：
#通过演示分别对前面交叉结果result1与softmax后的结果logits计算loss，验证结果如下：
#1）对于softmax_cross_entropy_with_logits后的结果求loss直接取均值
#2）对于softmax后的结果使用-tf.reduce_sum(labels*tf.log(logits_scaled))求loss
#3）对于softmax后的结果使用-tf.reduce_sum(labels*tf.log(logits_scaled),1)等同于
#softmax_cross_entropy_with_logits结果
#4）由于（1）和（3）可以推出对（3）进行求均值也可以得到正确的loss值，合并起来的公式为：
#tf.reduce_sum(-tf.reduce_sum(labels*tf.log(logits_scaled),1))=loss

loss=tf.reduce_sum(result1)
with tf.Session() as sess:
    print('实际的损失函数值：loss=',sess.run(loss))
    
#而对于rel3这种已经求的softmax的情况求loss，可以把公式进一步简化成：
#loss2=-tf.reduce_sum（labels*tf.log(logits_scaled))
lables=[[0,0,1],[0,1,0]]
loss2=-tf.reduce_sum(lables*tf.log(logits_scaled))
with tf.Session() as sess:
    print('与上述结果一致：loss2=',sess.run(loss2))
    
















