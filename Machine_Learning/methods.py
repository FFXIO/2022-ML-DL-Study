import numpy as np
from compute_utils import *
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def Linear_Regression_GD(x,y,b,w,learning_rate=0.01,epochs=10000):
    '''
    설명 : gradient descent를 통해서 b,w를 epoch만큼 업데이트 시키고 return w,b 
    x : [bsz,representation]
    y : [bsz,]
    w : [representation]
    b : [1]
    '''

    for _ in range(epochs) :
        hypothesis = w*x + b
        w_temp = (y - hypothesis)*x
        b_temp = y - hypothesis
        w = w + learning_rate*w_temp.sum()
        b = b + learning_rate*b_temp.sum()

    return w, b

def Linear_Regression(x,y,b,w,learning_rate=0.01,epochs=1000,batch_size=16):
    '''
    Linear Regression based on SGD
    input
        x       : [bsz,representation]
        y       : [bsz,1]
    parameters
        w       : [rep,number of class]
        b       : [1,number of class]
    output
        w       : [rep,number of class]
        b       : [1,number of class]
    '''

    for _ in range(epochs) :
        hypothesis = w * x + b
        w_temp = (y - hypothesis) * x
        b_temp = y - hypothesis
        w = w + learning_rate * np.random.choice(w_temp, min(len(x), batch_size), False).sum()
        b = b + learning_rate * np.random.choice(b_temp, min(len(x), batch_size), False).sum()
   
    return w,b

def Logistic_Regression(x,y,w,b,number_of_class,learning_rate=0.0001,epochs=300000,batch_size = 32):
    '''
    Logistic Regression based on SGD
    input 
        x       : [data size,rep]
        y       : [data_size,1]   
    parameters
        w       : [rep,number of class]
        b       : [1,number of class]
    output
        w       : [rep,number of class]
        b       : [1,number of class]
    '''
   
    return w,b

def SVM(x,y,w,b,number_of_class,C=30,learning_rate=0.001,epochs=10000):
    '''
    이진 linear SVM만을 가정
    input
        x       : [data size,rep]
        y       : [data size,1]
    parameters
        w       : [rep,number of class]
        b       : [1, number of class]
    output
        w       : [rep,number of class]
        b       : [1, number of class]
    '''
    
    return w,b

def Naive_Bayes(x,y,number_of_class):
    '''
    input
        x : [data size, rep]
        y : [data size, 1]
    output
        p_rep_status    : y의 상태에 따라 rep=1일 확률
        p_pos           : y=1인 데이터 발생 확률
    '''
    
    return x_threshold,p_rep_status,p_pos

def K_Means(data):
    '''
    input
        data : [data size, rep]
    '''
    number_of_centroids = 2
    return clusters, centroids


def Random_Forest(x_train, y_train, x_test, y_test):
    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(x_train, y_train)

    print('train result :', classifier.score(x_train, y_train))
    print('test result  :', classifier.score(x_test, y_test))

    confusion = confusion_matrix(y_test, classifier.predict(x_test))
    print('confusion matrix (test set) :\n', confusion)
    print('precision and recall (test set, 0 is positive) :\n')
    print('precistion =', confusion[0][0]/(confusion[0][0] + confusion[1][0]))
    print('recall =', confusion[0][0]/(confusion[0][0] + confusion[0][1]))
    print('report (correct answer) :\n', classification_report(y_test, classifier.predict(x_test)))