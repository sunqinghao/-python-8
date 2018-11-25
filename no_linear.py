from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris

def plot_samples(ax,samples):
    Y=samples[:,-1]
    Y=samples[:,-1]
    position_p=Y==1
    position_m=Y==-1
    ax.scatter(samples[position_p,0],samples[position_p,1],
        samples[position_p,2],marker='+',label='+',color='b')
    ax.scatter(samples[position_m,0],samples[position_m,1],
        samples[position_m,2],marker='^',label='-',color='y')

def creat_data_no_linear(n):
    np.random.seed(1)
    x_11 = np.random.randint(0, 100, (n, 1))
    x_12 = np.random.randint(0, 100, (n, 1))
    x_13 = 10 + np.random.randint(0, 10, (n, 1))
    x_21 = np.random.randint(0, 100, (n, 1))
    x_22 = np.random.randint(0, 100, (n, 1))
    x_23 = 20 - np.random.randint(0, 10, (n, 1))

    new_x_12 = x_12 * np.sqrt(2) / 2 - x_13 * np.sqrt(2) / 2
    new_x_13 = x_12 * np.sqrt(2) / 2 + x_13 * np.sqrt(2) / 2
    new_x_22 = x_22 * np.sqrt(2) / 2 - x_23 * np.sqrt(2) / 2
    new_x_23 = x_22 * np.sqrt(2) / 2 + x_23 * np.sqrt(2) / 2
    plus_samples = np.hstack([x_11, new_x_12, new_x_13, np.ones((n, 1))])
    minus_samples = np.hstack([x_21, new_x_22, new_x_23, -np.ones((n, 1))])
    samples = np.vstack([plus_samples, minus_samples])
    np.random.shuffle(samples)
    return samples
def perceptron(train_data,eta,w_0,b_0):
    x=train_data[:,:-1]
    y=train_data[:,-1]
    w=w_0
    b=b_0
    length=train_data.shape[0]
    step_num=0
    while True:
        i=0
        while i<length:
            step_num+=1
            x_i=x[i].reshape(x.shape[1],1)
            if step_num>=10000:
                print("fail,step_num=%d"%step_num)
                return
            if y[i]*(np.dot(np.transpose(w),x_i)+b)<=0:
                w+=eta*y[i]*x_i
                b+=eta*y[i]
                break
            else :
                i+=1
        if i==length:
            break
    return (w,b,step_num)
data=creat_data_no_linear(10)
perceptron(data,eta=0.1,w_0=np.zeros((3,1)),b_0=0)
