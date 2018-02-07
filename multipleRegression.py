#-*-coding:utf-8-*-

import tensorflow as tf
import numpy as np

# csv 파일 읽기
training_data = np.genfromtxt('./training_data_10000.csv', delimiter=',',  dtype='f', filling_values = 0)

# column name 제거
training_data = np.delete(training_data,0,0)

y_data=[]

# y 뽑아내기
for count in range(0, 10000):
	y_data.append(training_data[count][23])

# 원 데이터에서 y자리의 데이터 제거
x_data = np.delete(training_data,23,1)

# factor별 배열로 변환하기 위하여 transpose
x_data = x_data.transpose()

#           데이터 전처리 완료                                                                #
#---------------------------------------------------------------------------------------------#
#           데이터 분석 시작                                                                  #

W = tf.Variable(tf.random_uniform([1,23], -1.0, 1.0))

hypothesis = tf.matmul(W,x_data)

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.0000038)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(5000000):
	
	sess.run(train)
	if step % 200 == 0:
		print ("\n\n[",step,"]\n", "cost :\n", sess.run(cost), "\n\n W :\n", sess.run(W))

#기대값 : 1
case1 = [[2],[11],[155],[55],[71],[1.2],[1],[1],[1],[115],[75],[92],[194],[77],[65],[114],[13.3],[1],[0.7],[20],[17],[18],[1]]
result_mat1 = tf.matmul(W,case1)

# 기대값 : 1
case2 = [[2],[11],[155],[50],[77],[1.2],[1.2],[1],[1],[129],[79],[100],[196],[57],[76],[109],[12],[1],[0.7],[25],[14],[23],[1]]
result_mat2 = tf.matmul(W,case2)

# 기대값 : 0
case3 = [[2],[13],[145],[60],[87],[0.8],[0.7],[1],[1],[148],[94],[181],[225],[64],[75],[137],[15.2],[1],[0.6],[33],[37],[62],[1]]
result_mat3 = tf.matmul(W,case3)

# 기대값 : 0
case4 = [[1],[12],[165],[75],[92],[0.9],[0.9],[1],[1],[145],[89],[118],[209],[82],[48],[144],[17.2],[1],[1.3],[46],[79],[67],[1]]
result_mat4 = tf.matmul(W,case4)

print ("\ncase1 expected value : 1")
print ("\ncase1 : ",sess.run(result_mat1),"\n")
print ("\ncase2 expected value : 1")
print ("\ncase2 : ",sess.run(result_mat2),"\n")
print ("\ncase3 expected value : 0")
print ("\ncase3 : ",sess.run(result_mat3),"\n")
print ("\ncase4 expected value : 0")
print ("\ncase4 : ",sess.run(result_mat4),"\n")
