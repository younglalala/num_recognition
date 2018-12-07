import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np
from scipy import misc
from skimage import io
import matplotlib.image as mpimg


from train_sample import *
from test_sample import *
from onehott import *
from  tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)

class CNN:
    def __init__(self):
        self.x=tf.placeholder(dtype=tf.float32,shape=[None,28,28,1])
        self.y_=tf.placeholder(dtype=tf.float32,shape=[None,10])
        self.dp=tf.placeholder(tf.float32)

        self.conv1_w=tf.Variable(tf.random_normal([3,3,1,16],dtype=tf.float32,stddev=tf.sqrt(1/32)))
        self.conv1_b=tf.Variable(tf.zeros([16]))

        self.conv2_w=tf.Variable(tf.random_normal([3,3,16,32],dtype=tf.float32,stddev=tf.sqrt(1/64)))
        self.conv2_b=tf.Variable(tf.zeros([32]))

        # self.conv3_w = tf.Variable(tf.random_normal([3, 3, 32, 32], dtype=tf.float32, stddev=tf.sqrt(1 / 64)))
        # self.conv3_b = tf.Variable(tf.zeros([32]))

        # self.conv4_w = tf.Variable(tf.random_normal([3, 3, 64, 128], dtype=tf.float32, stddev=tf.sqrt(1 / 64)))
        # self.conv4_b = tf.Variable(tf.zeros([128]))

        self.fc1_w=tf.Variable(tf.random_normal([7*7*32,128],dtype=tf.float32,stddev=tf.sqrt(1/1024)))
        self.fc1_b=tf.Variable(tf.zeros([128]))

        self.fc2_w=tf.Variable(tf.random_normal([128,10],dtype=tf.float32,stddev=tf.sqrt(1/10)))
        self.fc2_b=tf.Variable(tf.zeros([10]))

    def forward(self):
        self.conv1=tf.nn.relu(tf.nn.conv2d(self.x,self.conv1_w,strides=[1,1,1,1],padding='SAME')+self.conv1_b)
        self.pool1=tf.nn.max_pool(self.conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

        self.conv2=tf.nn.relu(tf.nn.conv2d(self.pool1,self.conv2_w,strides=[1,1,1,1],padding='SAME')+self.conv2_b)
        self.pool2=tf.nn.max_pool(self.conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

        # self.conv3 = tf.nn.relu(
        #     tf.nn.conv2d(self.pool2, self.conv3_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv3_b)
        # self.pool3= tf.nn.max_pool(self.conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # self.conv4 = tf.nn.relu(
        #     tf.nn.conv2d(self.pool3, self.conv4_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv4_b)
        # self.pool4 = tf.nn.max_pool(self.conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # print(self.pool4)

        self.flat=tf.reshape(self.pool2,[-1,7*7*32])

        self.fc1=tf.nn.relu(tf.matmul(self.flat,self.fc1_w)+self.fc1_b)
        self.fc11=tf.nn.dropout(self.fc1,keep_prob=self.dp)
        self.out=tf.matmul(self.fc11,self.fc2_w)+self.fc2_b


    def backward(self):
        self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.out,labels=self.y_))
        self.optimizer=tf.train.AdamOptimizer(0.0006).minimize(self.loss)

        self.out_arg=tf.argmax(self.out,1)
        self.label_arg=tf.argmax(self.y_,1)

        self.acc=tf.reduce_mean(tf.cast(tf.equal(self.out_arg,self.label_arg),'float'))

if __name__=='__main__':
    net = CNN()
    net.forward()
    net.backward()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=5)
    # train_data = train_shuffle_batch(train_filename, [28, 28, 1], 100)
    test_data = test_shuffle_batch(test_filename, [28, 28, 1], 10000)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        sess.run(init)

        cout=0
        saver.restore(sess, "modle_save/train.cptk")

        x=[]
        y=[]
        for i in range(1):

            train_x,train_y=mnist.train.next_batch(50000)
            train_x=train_x.reshape([50000,28,28,1])

            for j in range(len(train_x)):
                label=np.where(train_y[j]>0.9)[0][0]
                print(train_x[j].shape)
                image=Image.fromarray(train_x[j])
                print()

                # image.save('/Users/wywy/Desktop/MNIST/{}_{}.jpg'.format(j,label))


                # misc.imsave('/Users/wywy/Desktop/MNIST/{}_{}.jpg'.format(j,label),image)

            # train_x, train_y =sess.run(train_data)
            # train_x=train_x/255-0.5
            # train_y = train_y.reshape([-1]).tolist()
            # train_y = one_hot(10, train_y)
            # print(train_y[0])

            # _,train_loss,train_acc=sess.run([net.optimizer,net.loss,net.acc],feed_dict={net.x:train_x,net.y_:train_y,net.dp:0.5})
            #
            # print('第{}次的误差为 {} ，精度为 {} '.format(i,train_loss,train_acc))
            # # x.append(i)
            # # y.append(train_loss)
            # # plt.plot(x,y,'red')
            # # plt.pause(0.01)
            # # plt.clf()
            #
            # if i%100==0:
            #     test_x,test_y=mnist.test.next_batch(10000)
            #     test_x = test_x.reshape([10000, 28, 28, 1])
            #
            #     # test_img=mpimg.imread('/Users/wywy/Desktop/xxxx/1348_8.jpg')
            #     # test_img=test_img.reshape([-1,28,28,1])
            #     # print(sess.run(net.out_arg,feed_dict={net.x:test_img,net.dp:1.}))
            #
            #
            #
            #     # test_x, test_y =sess.run(test_data)
            #     # test_y = test_y.reshape([-1]).tolist()
            #     # test_y = one_hot(9, test_y)
            #
            #     test_loss, test_acc = sess.run([net.loss, net.acc],
            #                                         feed_dict={net.x: test_x, net.y_: test_y,net.dp:1.})
            #
            #     print('第{}次测试集的误差为 {} ，精度为 {} '.format(i, test_loss, test_acc))
            #     # saver.save(sess, "modle_save/train.cptk")
            #
            #
            #



        # train_x1=train_x[0].reshape([28,28,1])
        # train_x1=train_x1.repeat([3],axis=2)
        # train_y=np.where(train_y[0]>0.7)[0].tolist()

        # misc.imsave ('/Users/wywy/Desktop/MNIST/{}_{}.jpg'.format(cout,train_y[0]),train_x1)
        # cout+=1


