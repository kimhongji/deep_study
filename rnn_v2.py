"""
2/8
패턴 지표 사용 (4개)
10일 데이터 넣고 10일 뒤 예측

STOCHASTIC
MOMENTUM
CCI
"""
#------------------------------------------------------------------------------
import matplotlib.pyplot as plt 
import tensorflow as tf 
import numpy as np
import talib as ta
tf.reset_default_graph() 
predict_len = 10  #몇일뒤 예측 날짜 인지 
train_index = 32 #몇번째 부터 넣을지(nan을 제외한 부분을 위해 )
#------------------------------------------------------------------------------
#데이터크기를 균일하게 보정
def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

#split, null 제거작업
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
#------------------------------------------------------------------------------
#파일을 불러와 x,y로 나눔
#a, raw_data = read_file('./data/lg.csv')
a, raw_data = read_file('./data/amore.csv')
#a, raw_data = read_file('./data/hyundai.csv')
#a, raw_data = read_file('./data/kt&g.csv')
#a, raw_data = read_file('./data/samsung_fire.csv')
#a, raw_data = read_file('./data/lotte_chemical.csv')
#a, raw_data = read_file('./data/lg_display.csv')

xy = np.array(raw_data)
xy = xy[::-1]               # reverse order (chronically ordered)
xy = MinMaxScaler(xy)       #안해주면 predict값이 모두 동일하게 나옴
x = xy                      #시가, 고가, 저가, 거래량, 종가
y = xy[:, [-1]]             #종가
#------------------------------------------------------------------------------
#x,y 행렬 전치 해줌 ( 그래야 talib가 먹힘 )
t_open = np.transpose(x[:,[0]])
t_high =  np.transpose(x[:,[1]])
t_low = np.transpose(x[:,[2]])
t_vol = np.transpose(x[:,[3]])
t_y = np.transpose(x[:,[4]])

#array로 변환
open = np.array(t_open[0],dtype='float')
high = np.array(t_high[0],dtype='float')
low = np.array(t_low[0],dtype='float')
volume = np.array(t_vol[0],dtype='float')
close = np.array(t_y[0],dtype='float')
#------------------------------------------------------------------------------
#주가데이터를 지표데이터로 변환(TA-Lib 이용)
slowk, slowd = ta.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0) #Stochastic
mom = ta.MOM(close,timeperiod=predict_len)
cci = ta.CCI(high,low,close,timeperiod=predict_len*2)
#------------------------------------------------------------------------------
#x를 지표데이터 15개 묶음으로 변경
x = np.c_[slowk, slowd, mom, cci] 
        
#***************************************************************************************************#여기서부터 사용하는 변수 다름   
 # train Parameters
R_seq_length = 10          #10일 데이치를 넣어 
R_after = 9               #그로부터 10일뒤 종가 예측
R_data_dim = 4           #지표로 변환하여 데이터 4개가 들어감
R_hidden_dim = 10         #lstm에서 output은 10개가 나옴
R_output_dim = 1          #FC를 통해 10개에서 1개로
R_learning_rate = 0.05   #학습률
R_iterations = 1000        #반복 횟수

#X,Y
R_x = x[train_index: -1]            #nan이 아닌 부분부터
R_y = y[train_index: -1]          #x시작에 맞게 nan이 아닌 부분부터

#------------------------------------------------------------------------------
# build a dataset(talib) dataX에는 10일치 데이터, dataY는 그로부터 10일 뒤 종가 데이터 저장
R_dataX = []
R_dataY = []
for R_i in range(0, len(R_y) - R_seq_length - R_after):
    _x = R_x[R_i:R_i + R_seq_length]
    _y = R_y[R_i + R_seq_length + R_after]  # Next close price
    R_dataX.append(_x)
    R_dataY.append(_y)   
    
# train/test split (talib) 
R_train_size = int(len(R_dataY) * 0.7)
R_trainX, R_testX = np.array(R_dataX[0:R_train_size]), np.array(R_dataX[R_train_size:len(R_dataX)-500])
R_trainY, R_testY = np.array(R_dataY[0:R_train_size]), np.array(R_dataY[R_train_size:len(R_dataY)-500])
#------------------------------------------------------------------------------
# input place holders
R_X = tf.placeholder(tf.float32, [None, R_seq_length, R_data_dim])
R_Y = tf.placeholder(tf.float32, [None, 1])

# build a LSTM network
R_cell = tf.contrib.rnn.BasicLSTMCell(num_units=R_hidden_dim, state_is_tuple=True, activation=tf.tanh) #lstm 셀 생성
R_outputs, R_states = tf.nn.dynamic_rnn(R_cell, R_X, dtype=tf.float32) #cell output node
R_Y_pred = tf.contrib.layers.fully_connected(R_outputs[:, -1], R_output_dim, activation_fn=None)  #가장 마지막 lstm output(10개)을 FC를 통해 한개로 출력

# cost/loss
R_loss = tf.reduce_sum(tf.square(R_Y_pred - R_Y))

# optimizer
R_optimizer = tf.train.AdamOptimizer(R_learning_rate)
R_train = R_optimizer.minimize(R_loss)
#------------------------------------------------------------------------------
#session
R_init = tf.global_variables_initializer()
R_sess = tf.Session()
R_sess.run(R_init)

# Training step
for R_i in range(R_iterations):
    _R, R_step_loss, R_pred = R_sess.run([R_train, R_loss, R_Y_pred], feed_dict={R_X: R_trainX, R_Y: R_trainY})
    if (R_i + 1) % 100 == 0:
        print("R_loss: {}".format(R_step_loss))

# Test step
RNN_prediction = R_sess.run(R_Y_pred, feed_dict={R_X: R_testX}) #예측값

#정확도 측정을 위해 값 복사
R_prediction = np.array(RNN_prediction[0:len(RNN_prediction)]) 
R_target = np.array(R_testY[0:len(R_testY)]) #실제값
#------------------------------------------------------------------------------
"""
#선형 결과를 0,1로 변환(이전 종가보다 올랐는지, 내려갔는지로 판단)
for i in range(len(R_target)) :
    if(R_target[i:i+1] < R_target[i+1:i+2]) :  #전날의 종가에 비해 올랐을 때
        R_target[i:i+1] = 0
    else :                                    #떨어졌을 때
        R_target[i:i+1] = 1

for i in range(len(R_prediction)) :
    if(R_prediction[i:i+1] < R_prediction[i+1:i+2]) : 
        R_prediction[i:i+1] = 0
    else :                                    
        R_prediction[i:i+1] = 1
"""

#10일 전 종가와 비교하여 0,1로 변환      
test_close = np.array(R_dataY[R_train_size-10:len(R_dataY)-10]) #10인날의 종가

for i in range(len(R_target)) :
    if(R_target[i:i+1] < test_close[i:i+1]) :  #전날의 종가에 비해 올랐을 때
        R_target[i:i+1] = 0
    else :                                    #떨어졌을 때
        R_target[i:i+1] = 1

for i in range(len(R_prediction)) :
    if(R_prediction[i:i+1] < test_close[i:i+1]) : 
        R_prediction[i:i+1] = 0
    else :                                    
        R_prediction[i:i+1] = 1

    
#정확도 
R_is_correct = tf.equal(R_prediction, R_target)
R_accuracy = tf.reduce_mean(tf.cast(R_is_correct, tf.float32))

print('>>RNN Accuracy: %.2f' % R_sess.run(R_accuracy*100, feed_dict = {R_X: R_testX, R_Y: R_testY}))
#------------------------------------------------------------------------------
#그래프로 표현

plt.plot(R_pred, label='pred')
plt.plot(R_trainY, label='real')
plt.xlabel("Time Period")
plt.ylabel("Stock Price")
plt.legend()
plt.show()

plt.plot(RNN_prediction, label='pred')
plt.plot(R_testY, label='real')
plt.xlabel("Time Period")
plt.ylabel("Stock Price")
plt.legend()
plt.show()

plt.plot(R_prediction[140:180], label='pred')
plt.plot(R_target[140:180], label='real')
plt.xlabel("Time Period")
plt.ylabel("Stock Price")
plt.legend()
plt.show()










































