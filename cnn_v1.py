# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 19:44:07 2018

@author: yooop
"""

#14년치 데이터 가지고 해보는 중
# CNN on stock
import tensorflow as tf
import talib as ta
import numpy as np
tf.reset_default_graph()
tf.set_random_seed(777)  # reproducibility

#데이터 불러오기ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
def del_null_line(data):
    output=[]
    date=[]
    for i in data:
        i=i.split(",")
        try:
            date.append(i[0])
            output.append(list(map(float,i[1:])))
        except ValueError as e:
            print(e)
    #print(date)    
    return date,output

def read_file(filename):
    f = open(filename, 'r').read()
    data = f.split('\n')[:-2]
    raw_data=[]
    
    info=data[0].split(",")
    date,raw_data=del_null_line(data)
    
    return info, raw_data

a, raw_data = read_file('../data/lg.csv')
xy = np.array(raw_data)


#data processingㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
#xy = np.loadtxt('hyundai.csv', delimiter=',')
#xy = MinMaxScaler(xy)
x_data = np.c_[xy[:, 0:-2]/1000, xy[:,[-2]]/1000]
y_data = xy[:, [-1]]/1000
t_open = np.transpose(x_data[:,[0]])
t_high =  np.transpose(x_data[:,[1]])
t_low = np.transpose(x_data[:,[2]])
t_vol = np.transpose(x_data[:,[3]])
t_y = np.transpose(y_data)
open = np.array(t_open[0],dtype='float')
high = np.array(t_high[0],dtype='float')
low = np.array(t_low[0],dtype='float')
volume = np.array(t_vol[0],dtype='float')
close = np.array(t_y[0],dtype='float')
b_upper, b_middle, b_lower = ta.BBANDS(close) #bollinger ban
ma = ta.MA(close,timeperiod=10)         #moving average
dmi = ta.DX(high,low,close,timeperiod=10)#direct movement index
macd, macdsignal, macdhist = ta.MACD(close,fastperiod=10, slowperiod=14,signalperiod=4)
slowk, slowd = ta.STOCH(high, low, close, fastk_period=10, slowk_period=5, slowk_matype=0, slowd_period=5, slowd_matype=0)
#지표 5개 추가 SMA / WMA / MOM / RSI / CCI
mom = ta.MOM(close,timeperiod=10)
rsi = ta.RSI(close,timeperiod=10)
cci = ta.CCI(high,low,close,timeperiod=10)
sma = ta.SMA(close,timeperiod=10)#30으로 나와있긴 하다
wma = ta.WMA(close,timeperiod=10)#30으로 나와있
x_data = np.c_[b_upper, b_middle, b_lower,ma,dmi,macd, macdsignal, macdhist,slowk, slowd,mom,rsi,cci,sma,wma] 
x_data = x_data[17:-1]
y_data=  y_data[17:-1] 
train_size = int(len(y_data) * 0.7)#292개 가지고 학습
test_size = int(len(y_data) * 0.7)#117 test size이건데
#그냥 학습하고 남은 부분 부터 끝까지 test로 사용하고 있다 바꿔야 할 듯
trainX, testX = np.array(x_data[0:train_size]), np.array(x_data[test_size:len(x_data)])
trainY, testY = np.array(y_data[0:train_size]), np.array(y_data[test_size:len(y_data)])
test_cl =np.array(close[test_size:len(y_data)])
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

# hyper parameters
learning_rate = 0.0005

# input place holders
X = tf.placeholder(tf.float32, [10, 15])#10*10
X_img = tf.reshape(X, [-1, 1, 15, 10])#292일 10개 종목 data를 이미지 처럼
Y = tf.placeholder(tf.float32, [None,1])#마지막 결과 Y 하나로
K =tf.placeholder(tf.int32)

#layer1ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
W1 = tf.Variable(tf.random_normal([2, 2, 10, 200], stddev=0.01))#10 -> 100
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 5, 5, 1], padding='SAME')#pooling 끝나면 10/5 -> 2로
print(L1)

#layer2ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
W2 = tf.Variable(tf.random_normal([2, 2, 200, 400], stddev=0.01))#100 -> 200로 만들어 주는 필터
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],strides=[1, 3, 3, 1], padding='SAME')#pooling 끝나면 2/2 -> 1로
print(L2)
L2_flat = tf.reshape(L2, [-1,400])#fully connected로 만들어주기
print(L2_flat)

# Final FC 200inputs -> 1 outputsㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
W3 = tf.get_variable("W3", shape=[400,1],initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([1]))
logits = tf.matmul(L2_flat,W3) + b#1*512 - 512*1 해서 1*1

# define cost/loss & optimizer
real_Y = Y[K+14]#10일치 하고 5일 뒤의 Y값으로 예측
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=real_Y))
cost = tf.reduce_mean(tf.square((real_Y - logits)))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


#graph trainㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(10):
    for i in range(2270):
        x_d = trainX[i:i+10,:]
        sess.run(optimizer, feed_dict={X: x_d, Y:trainY,K:i})
    
    if (step + 1) % 5 == 0:
      print(step + 1, sess.run(cost, feed_dict={X: x_d, Y:trainY ,K:i}))

print("done")

#graph testㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
num = 0;
#result = np.array()
result = list(range(965))
for j in range(965):
    x_d = testX[j:j+10,:]
    prediction = sess.run(logits,feed_dict={X:x_d})
    #target = sess.run(Y,feed_dict={Y:testY,K:j})
    if(j+1)%100 == 0:
        print(prediction)
        print(testY[j+14])
    y_onehot = test_cl[j+9] - testY[j+14] # 실제 test 끼리의 차이
    y_onehot1= test_cl[j+9] - prediction # 예측한 차이로 당락 만을 예측 (폭 은 예상 X))
    if y_onehot>0 and y_onehot1>0:#올라가면
       result[j] = 1
       num = num+1
    elif y_onehot<0 and y_onehot1<0:#내려가면
       result[j] = 1
       num = num+1
    else:
        result[j] = 0

         
num = num/965
print('정확도: %.2f' %(num*100))
#open 종가로 바꾸고
#맟추면 1 틀리면 0으로 array만들기