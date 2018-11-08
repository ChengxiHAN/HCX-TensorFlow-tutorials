# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 20:02:37 2018

@author: LDog
"""

#2018-11-08 HCX at WHU LIESMARS 220 Computer Room
#更简单地保存检查点：1）tf.train.MonitiredTrainingSession函数
#该函数可以直接实现保存及载入检查点模型的文件，与 
#2）saver.save(sess,savedir+'linermodel.cpkt',global_step=epoch)法不同-》需要按照循环步数来保存
#而1）是按照训练时间来保存的，通过制定save_checkpoint_secs参数的具体秒数，来设置每训练多久保存一次检查点

#演示使用MonitoredTrainingSession函数来自动管理检查点文件

import tensorflow as tf
tf.reset_default_graph()
global_step=tf.train.get_or_create_global_step()
step=tf.assign_add(global_step,1)

#设置检查点路径为log/checkpoints
with tf.train.MonitoredTrainingSession(checkpoint_dir='log/checkpoints',save_checkpoint_secs=2) as sess:
    print(sess.run([global_step]))
    while not sess.should_stop():   #启用死循环
        i=sess.run(step)
        print(i)
        
        
#停止程序之后，可以再重新开始程序，发现程序从上一次结尾的程序开始继续执行
    
#TIPS:
#    1。如果不设置save_checkpoint_secs参数，默认的保存时间间隔为10分钟，
#    这种按照时间保存的模式更适用于大型数据集来训练复杂模型的情况
#    2.使用该方法时，必须定义global———step变量，否则会报错
    
    