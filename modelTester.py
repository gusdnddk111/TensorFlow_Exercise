#-*-coding:utf-8-*-

import tensorflow as tf
import numpy as np

# csv 파일 읽기
test_data = np.genfromtxt('./test_data_2000.csv', delimiter=',',  dtype='f', filling_values = 0)

y_data=[]

# y 뽑아내기
for count in range(0, 2000):
	y_data.append(test_data[count][23])

# 원 데이터에서 y자리의 데이터 제거
x_data = np.delete(test_data,23,1)

# factor별 배열로 변환하기 위하여 transpose
x_data = x_data.transpose()

#           데이터 전처리 완료                                                                #
#---------------------------------------------------------------------------------------------#
#           데이터 테스트 시작                                                                  #

W = [[ 1.2500000e-01, -4.7157090e-02, 1.1042322e-02, -3.8343261e-03, 4.0016795e-04, 5.7907281e-03, 5.8422051e-03, -7.3566414e-02, 2.6876789e-01, -3.1539216e-03, -5.3516089e-04, -1.9702085e-03, -1.3583256e-03, -6.2704447e-04, 1.7578177e-03, -5.2688376e-04, 1.3567152e-02, -1.8962380e-02, -1.8374656e-01, 4.6890430e-04, -4.1780830e-03, -1.8009357e-04, -1.3078620e-02]]

sess = tf.Session()

result_mat = tf.matmul(W, x_data)
result = sess.run(result_mat)

count = 0

for step in range(0,1999):
	if result[0, step] > 0.5:
		result[0, step] = 1
	else:
		result[0, step] = 0

	if(result[0,step] == y_data[step]):
		count += 1

print ("\naccuracy : %.1f"	%((count/2000.0)*100),"%")
