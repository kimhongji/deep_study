#필요한 lib import
'''
1/16
hongji
---------------
'''

import tensorflow as tf
import numpy as np
import talib as ta  
import matplotlib.pyplot as plt

############################################
#이부분 부터 -----------복사 복붙해서 사용 하면 될듯
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
##############################################
#xy = np.loadtxt('../data/hyundai.csv', delimiter=',')

#일단 일반화해서 nomal 을 못하겠어서 이런식으로 내가 직접 값을 비슷하게 해줌 
x_data = np.c_[xy[:, 0:-2]/1000, xy[:,[-2]]/1000000] #numpy로 데이터 합치기 
y_data = xy[:, [-1]]/1000

#transpose data (high, low , close)------
t_open = np.transpose(x_data[:,[0]])
t_high =  np.transpose(x_data[:,[1]])#x,y 행렬 전치 해줌 ( 그래야 talib가 먹힘 )
t_low = np.transpose(x_data[:,[2]])
t_vol = np.transpose(x_data[:,[3]])
t_y = np.transpose(y_data)

open = np.array(t_open[0],dtype='float')
high = np.array(t_high[0],dtype='float')
low = np.array(t_low[0],dtype='float')
volume = np.array(t_vol[0],dtype='float')
close = np.array(t_y[0],dtype='float')
#-----------------------------------------

#TA- data ( 5종목의 각각 3,1,1,3,2개의 값을 가짐)
#변수들 선언 
seq_len = 10    #학습 1set 에 들어가는 일 수 
predict_len = 30  #몇일뒤 예측 날짜 인지 
learning_rate = 0.01
train_index = 65 #몇번째 부터 넣을지(nan을 제외한 부분을 위해 )

#TA-Lib data 전처기 과정--------------------------------
b_upper, b_middle, b_lower = ta.BBANDS(close) #bollinger ban
ma = ta.MA(close,timeperiod=predict_len)         #moving average
dmi = ta.DX(high,low,close,timeperiod=predict_len)#direct movement index
macd, macdsignal, macdhist = ta.MACD(close,fastperiod=predict_len, slowperiod=predict_len*2,signalperiod=4) #moving average convergence/divergence
slowk, slowd = ta.STOCH(high, low, close, fastk_period=predict_len, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0) #Stochastic
mom = ta.MOM(close,timeperiod=predict_len)
rsi = ta.RSI(close,timeperiod=predict_len*2) #상대강도지수(0으로 갈수록 상향이라는)
cci = ta.CCI(high,low,close,timeperiod=predict_len*2)
sma = ta.SMA(close,timeperiod=predict_len)#30으로 나와있긴 하다
wma = ta.WMA(close,timeperiod=predict_len)#30으로 나와있

#----------------------------------------------------

x_data = np.c_[b_upper, b_middle, b_lower,ma,dmi,macd, macdsignal, macdhist,slowk, slowd,mom,rsi,cci,sma,wma] 
#바꾼ta-lib 데이터를 x_data 로 합쳐준다. 총 10개의 지표 값이 나온다(당일에 관해)
close = close[train_index:-1-predict_len]
x_data = x_data[train_index:-1-predict_len]#예측 가능한 부분 까지 사용 하기 위해 (predict_len 일뒤의 y 값이 존재하는 )
y_data = y_data[train_index+predict_len:-1]#nan 이 아니었던 부분만 쓰기 위해 14일 부터 학습의 10 일과 그 부터 7일뒤 y 값

train_size = int(len(y_data) * 0.7)
test_size = int(len(y_data) * 0.7)
testC =  np.array(close[test_size:len(x_data)])
trainX, testX = np.array(x_data[0:train_size]), np.array(x_data[test_size:len(x_data)])
trainY, testY = np.array(y_data[0:train_size]), np.array(y_data[test_size:len(y_data)])

###########
# 신경망 모델 구성

###########

#X = tf.placeholder(tf.float32, [None, 4])
X = tf.placeholder(tf.float32, [None, 15])#10개의 값이 10일치 가 1-set
Y = tf.placeholder(tf.float32, [None, 1])

#총 3개의 hidden layer 와 입력 출력 layer 가 있다
#layer 1 : 입력 레이어 열의 개수가 4개로 들어가 10개의 전달 값을 갖는다.
#W1 = tf.Variable(tf.random_uniform([4, 10],-1.,1.))
W1 = tf.Variable(tf.random_uniform([15, 18],-1.,1.))
b1 = tf.Variable(tf.random_normal([18]))
L1 = tf.nn.relu(tf.matmul(X, W1)+b1)
#L1 = tf.nn.dropout(L1, 0.8)

#layer 2 : 은닉1 레이어 열의 개수가 10개로 들어가 8개의 전달 값을 갖는다. 
W2 = tf.Variable(tf.random_normal([18, 10]))
b2 = tf.Variable(tf.random_normal([10]))
L2 = tf.nn.relu(tf.matmul(L1, W2)+b2)
#L2 = tf.layers.dropout(L2, 0.2)#overfitting 방지 

#layer 4 : 은닉3 레이어 열의 개수가 6개로 들어가 4개의 전달 값을 갖는다. 
W4 = tf.Variable(tf.random_normal([10, 6]))
b4 = tf.Variable(tf.random_normal([6]))
L4 = tf.nn.relu(tf.matmul(L2, W4)+b4)
#L4 = tf.layers.dropout(L4, 0.2)

#layer 5 :출력 레이어 열의 개수가 4개로 들어가 1개의 전달 값을 갖는다. 
W5 = tf.Variable(tf.random_normal([6, 1]))
b5 = tf.Variable(tf.random_normal([1]))
#L5 = tf.nn.relu(tf.matmul(L4, W5)+b5)
model = tf.add(tf.matmul(L4, W5),b5)

#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
cost = tf.reduce_mean(tf.square((Y - model)))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#########
# 신경망 모델 학습
######

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(10000):
    sess.run(optimizer, feed_dict={X: trainX, Y: trainY})
    
    if (step + 1) % 1000 == 0:
       print(step + 1, sess.run([cost], feed_dict={X: trainX, Y: trainY}))
        
print('최적화 끝')

#########
# 결과 확인
#선형데이터로 예측해 +,- 로 확인 
######
prediction =sess.run(model, feed_dict={X: testX}) #예측값
target = sess.run(Y, feed_dict={Y: testY}) #실제값

y_onehot = tf.sign(testC[:] - target[predict_len:,[-1]]) # 실제 test 끼리의 차이 open 과 close 값으로 계산 
y_onehot1= tf.sign(testC[:] - prediction[predict_len:,[0]]) # 예측한 차이로 당락 만을 예측 (폭 은 예상 X)5
is_correct = tf.equal(y_onehot, y_onehot1)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: testX, Y: testY}))

plt.plot(testY) 
plt.plot(prediction)
plt.xlabel("Time Period") 
plt.ylabel("Stock Price") 
plt.show()







