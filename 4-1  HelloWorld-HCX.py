# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 10:47:47 2018

@author: LDog
"""
#2018-11-08 HCX at WHU LIESMARS 220 Computer Room

import tensorflow as tf

#-----------测试tensorflow是否安装好，第一个小例子#-----------
hello=tf.constant('Hello World!')
with tf.Session() as sess:   #with 沿用了python的with用法，即当程序结束后会自动关系session，不需要写close
    print(sess.run(hello))
    
#-----------演示注入机制-----------
#使用注入机制，将具体的实参注入到相应的placeholder中。
#feed只在调用它的方法有效，方法结束后feed就会自动消失
a=tf.placeholder(tf.int16)
b=tf.placeholder(tf.int16)
add=tf.add(a,b) #形参
mul=tf.multiply(a,b)
#sess.InteractiveSession()  #交互式session的方式，一般在Jupyter环境下使用较多
with tf.Session() as sess:
    #计算具体的数值
    print('相加： %i' %sess.run(add,feed_dict={a:3,b:4}))
    print('相乘： %i' %sess.run(mul,feed_dict={a:3,b:4}))
    #使用注入机制获取节点
    print(sess.run([mul,add],feed_dict={a:3,b:4})) 
    
    
    
with tf.Session() as sess:
#    with tf.device('/cpu：1'): #在gpu版本的TensorFlow上可以指定gpu计算
        print('相加： %i' %sess.run(add,feed_dict={a:3,b:4}))
        print('相乘： %i' %sess.run(mul,feed_dict={a:3,b:4}))
#        使用注入机制获取节点
        print(sess.run([mul,add],feed_dict={a:3,b:4}))
        
    
    
    
    
    
    
    
    
    
    

