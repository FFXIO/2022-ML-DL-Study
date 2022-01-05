from compute_utils import *
from arguments import get_arguments
from methods import *
from sklearn.datasets import load_iris, make_blobs, load_breast_cancer
import argparse
import numpy as np
import collections

def main():
    # argument parsing
    args = argparse.ArgumentParser()
    args = get_arguments()

    if args.methods=='Linear_Regression':
        # y = 2x+5
        x = np.array([2,6,9])
        y = np.array([9,17,23])
        w = np.random.uniform(-1,1,1) # np.random.uniform(low, high, size)
        b = np.random.uniform(-1,1,1)
        w,b = globals()[args.methods](x,y,w,b)

        # correct answer : w = 2, b = 5
        x_test = np.array(range(0, 20))
        y_test = x_test*2+5

        # predict y by Linear_Regression
        w_linear,b_linear = Linear_Regression(x, y, b, w, learning_rate=0.01, epochs=1000, batch_size=16)
        y_predict_linear = x_test*w_linear + b_linear

        # predict y by Linear_Regression_GD
        w_linear_GD, b_linear_GD = Linear_Regression_GD(x,y,b,w,learning_rate=0.01,epochs=10000)
        y_predict_linear_GD = x_test * w_linear_GD + b_linear_GD

        # Least square estimation
        LSE = ((y_predict_linear - y_test) ** 2) / 2
        print("Linera_Regression : calculate accuracy by LSE :", LSE.sum())
        LSE_GD = ((y_predict_linear_GD - y_test) ** 2) / 2
        print("Linera_Regression_GD : calculate accuracy by LSE :", LSE_GD.sum())

        # Maximum Likelihood Estimation


    elif args.methods=='Logistic_Regression':
        x,y = load_iris(return_X_y=True)
        number_of_class = len(collections.Counter(y))
        x_train,y_train,x_test,y_test = divide_dataset(x,y)
        w = np.random.uniform(-1,1,(x.shape[1],number_of_class))
        b = np.random.uniform(-1,1,(number_of_class))
        w,b = globals()[args.methods](x_train,y_train,w,b,number_of_class)

        '''
        PREDICTION
        '''

    elif args.methods=='SVM':
        x,y = make_blobs(n_samples=150,centers=2,random_state=20)
        y = np.sign(y-0.5).astype(np.int64)
        number_of_class = len(collections.Counter(y))
        x_train,y_train,x_test,y_test = divide_dataset(x,y)
        w = np.ones((x.shape[1]))
        b = np.ones((1))
        w,b = globals()[args.methods](x_train,y_train,w,b,number_of_class)

        '''
        PREDICTION
        '''

    elif args.methods=='Naive_Bayes':
        data = load_breast_cancer()
        x = data['data']
        y = data['target']
        x_train,y_train,x_test,y_test = divide_dataset(x,y)
        number_of_class = len(collections.Counter(y))
        x_threshold,p_rep_status,p_pos = globals()[args.methods](x_train,y_train,number_of_class)

        '''
        PREDICTION
        '''
      
    elif args.methods =='K_Means':
        x1 = np.random.uniform(-5,0,100)
        y1 = np.random.uniform(-5,0,100)
        x2 = np.random.uniform(5,10,50)
        y2 = np.random.uniform(5,10,50)
        data1 = np.vstack([x1,y1]).T
        data2 = np.vstack([x2,y2]).T
        data = np.concatenate([data1,data2],axis=0)
        clusters,centroids =globals()[args.methods](data)
        '''
        PREDICTION
        '''

if __name__=='__main__':
    main()