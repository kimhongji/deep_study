# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 20:11:44 2018

@author: kimhongji

code : scikit learn (Linear regression example )
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

#데이터 갸져오기 
diabetes = datasets.load_diabetes()

#diabetes 의 총 10개의 데이터중 2번째 데이터만 가져옴
diabetes_X = diabetes.data[:, np.newaxis, 2]

#뒤에서 -20 까지의 데이터 를 train 
#뒤에서 20 개의 데이터를 test 해서 학습 and 검증 
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

diabetes_Y_train = diabetes.target[:-20]
diabetes_Y_test = diabetes.target[-20:]

#linear regression API 이용 
regr = linear_model.LinearRegression()

#.fit 함수 이용해 학습 
regr.fit(diabetes_X_train, diabetes_Y_train)

#predict 함수 이용해 예측값 출력 
diabetes_Y_pred = regr.predict(diabetes_X_test)

print('coefficients : \n', regr.coef_)

print("Mean squard error : %.2f" % mean_squared_error(diabetes_Y_test, diabetes_Y_pred))

print('Variance score : %.2f' % r2_score(diabetes_Y_test, diabetes_Y_pred))

plt.scatter(diabetes_X_test,diabetes_Y_test, color='black' )
plt.plot(diabetes_X_test,diabetes_Y_pred, color = 'blue',linewidth=3)

plt.xticks()
plt.yticks()
plt.show()
