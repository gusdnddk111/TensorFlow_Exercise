#-*-coding:utf-8-*-

import tensorflow as tf
from numpy import genfromtxt

training_data = genfromtxt('./training_data.csv', delimiter=',')

print (training_data[0][1])

x1_data=[]
x2_data=[]
x3_data=[]
x4_data=[]
x5_data=[]
x6_data=[]
x7_data=[]
x8_data=[]
x9_data=[]
x10_data=[]
x11_data=[]
x12_data=[]
x13_data=[]
x14_data=[]
x15_data=[]
x16_data=[]
x17_data=[]
x18_data=[]
x19_data=[]
x20_data=[]
x21_data=[]
x22_data=[]
x23_data=[]
x24_data=[]

y_data=[]

for count in range(1,100):
	x1_data.append(training_data[count][0])
	x2_data.append(training_data[count][1])
	x3_data.append(training_data[count][2])
	x4_data.append(training_data[count][3])
	x5_data.append(training_data[count][4])
	x6_data.append(training_data[count][5])
	x7_data.append(training_data[count][6])
	x8_data.append(training_data[count][7])
	x9_data.append(training_data[count][8])
	x10_data.append(training_data[count][9])
	x11_data.append(training_data[count][10])
	x12_data.append(training_data[count][11])
	x13_data.append(training_data[count][12])
	x14_data.append(training_data[count][13])
	x15_data.append(training_data[count][14])
	x16_data.append(training_data[count][15])
	x17_data.append(training_data[count][16])
	x18_data.append(training_data[count][17])
	x19_data.append(training_data[count][18])
	x20_data.append(training_data[count][19])
	x21_data.append(training_data[count][20])
	x22_data.append(training_data[count][21])
	x23_data.append(training_data[count][22])	
	x24_data.append(training_data[count][23])

W1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W3 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W4 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W5 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W6 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W7 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W8 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W9 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W10 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W11 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W12 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W13 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W14 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W15 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W16 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W17 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W18 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W19 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W20 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W21 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W22 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W23 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W24 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

hypothesis = W1 * x1_data + W2 * x2_data + W3 * x3_data + W4 * x4_data + W5 * x5_data + W6 * x6_data + W7 * x7_data + W8 * x8_data + W9 * x9_data + W10 * x10_data + W11 * x11_data + W12 * x12_data + W13 * x13_data + W14 * x14_data + W15 * x15_data + W16 * x16_data + W17 * x17_data + W18 * x18_data + W19 * x19_data + W20 * x20_data + W21 * x21_data + W22 * x22_data + W23 * x23_data + W24 * x24_data + b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train)
	if step % 20 == 0:
        print (step, sess.run(cost), sess.run(W1), sess.run(W2), sess.run(b))
