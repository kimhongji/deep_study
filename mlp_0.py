#필요한 lib import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')

x_data = np.c_[xy[:, 0:-2]/10, xy[:,[-2]]/100000] 
y_data = xy[:, [-1]]/10


train_size = int(len(y_data) * 0.7)
test_size = len(y_data) - train_size
trainX, testX = np.array(x_data[0:train_size]), np.array(x_data[train_size:len(x_data)])
trainY, testY = np.array(y_data[0:train_size]), np.array(y_data[train_size:len(y_data)])

#########
# 신경망 모델 구성
########

X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.float32, [None, 1])

#총 3개의 hidden layer 와 입력 출력 layer 가 있다
#layer 1 : 입력 레이어 열의 개수가 4개로 들어가 10개의 전달 값을 갖는다. 
W1 = tf.Variable(tf.random_uniform([4, 10],-1.,1.))
b1 = tf.Variable(tf.random_normal([10]))
L1 = tf.nn.relu(tf.matmul(X, W1)+b1)
#L1 = tf.nn.dropout(L1, 0.8)

#layer 2 : 은닉1 레이어 열의 개수가 10개로 들어가 8개의 전달 값을 갖는다. 
W2 = tf.Variable(tf.random_normal([10, 8]))
b2 = tf.Variable(tf.random_normal([8]))
L2 = tf.nn.relu(tf.matmul(L1, W2)+b2)
#L2 = tf.nn.dropout(L2, 0.8)

#layer 3 : 은닉3 레이어 열의 개수가 8개로 들어가 6개의 전달 값을 갖는다. 
W3 = tf.Variable(tf.random_normal([8, 6]))
b3 = tf.Variable(tf.random_normal([6]))
L3 = tf.nn.relu(tf.matmul(L2, W3)+b3)
#L3 = tf.nn.dropout(L3, 0.8)

#layer 4 : 은닉3 레이어 열의 개수가 6개로 들어가 4개의 전달 값을 갖는다. 
W4 = tf.Variable(tf.random_normal([6, 4]))
b4 = tf.Variable(tf.random_normal([4]))
L4 = tf.nn.relu(tf.matmul(L3, W4)+b4)

#layer 5 :출력 레이어 열의 개수가 4개로 들어가 1개의 전달 값을 갖는다. 
W5 = tf.Variable(tf.random_normal([4, 1]))
b5 = tf.Variable(tf.random_normal([1]))
#L5 = tf.nn.relu(tf.matmul(L4, W5)+b5)
model = tf.add(tf.matmul(L4, W5),b5)

#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
cost = tf.reduce_mean(tf.square((Y - model)))
optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

#########
# 신경망 모델 학습
######

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(10000):
    sess.run(optimizer, feed_dict={X: trainX, Y: trainY})
    
    if (step + 1) % 1000 == 0:
        print(step + 1, sess.run(cost, feed_dict={X: trainX, Y: trainY}))

print('최적화 끝')

#########
# 결과 확인
#0,1 : 하락 , 1,0 : 상승 
######
prediction =sess.run(model, feed_dict={X: testX})
target = sess.run(Y, feed_dict={Y: testY})

y_onehot = tf.sign(testX[:,[0]] - testY[:,[-1]]) # 실제 test 끼리의 차이
y_onehot1= tf.sign(testX[:,[0]] - prediction[:,[0]]) # 예측한 차이로 당락 만을 예측 (폭 은 예상 X)
is_correct = tf.equal(y_onehot, y_onehot1)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: testX, Y: testY}))


plt.plot(testY) 
plt.plot(prediction)
plt.show()