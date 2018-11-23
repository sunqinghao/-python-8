# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris

'''
def creat_data(n):
    np.random.seed(1)
    x_11 = np.random.randint(0,100,(n,1))
    x_12 = np.random.randint(0, 100, (n, 1))
    x_13 = 20+np.random.randint(0, 10, (n, 1))
    x_21 = np.random.randint(0, 100, (n, 1))
    x_22 = np.random.randint(0, 100, (n, 1))
    x_23 = 10-np.random.randint(0, 10, (n, 1))

    new_x_12=  x_12 * np.sqrt(2) / 2 - x_13 * np.sqrt(2) / 2
    new_x_13 = x_12 * np.sqrt(2) / 2 + x_13 * np.sqrt(2) / 2
    new_x_22 = x_22 * np.sqrt(2) / 2 - x_23 * np.sqrt(2) / 2
    new_x_23 = x_22 * np.sqrt(2) / 2 + x_23 * np.sqrt(2) / 2

    plus_samples=np.hstack([x_11,new_x_12,new_x_13,np.ones((n,1))])
    minus_samples=np.hstack([x_21,new_x_22,new_x_23,-np.ones((n,1))])
    samples=np.vstack([plus_samples,minus_samples])
    np.random.shuffle(samples)
    return samples

def plot_samples(ax,samples):
    Y=samples[:,-1]
    Y = samples[:, -1]
    position_p=Y==1
    position_m=Y==-1
    ax.scatter(samples[position_p,0],samples[position_p,1],samples[position_p,2],marker='+',label='+',color='b')
    ax.scatter(samples[position_m, 0], samples[position_m, 1], samples[position_m, 2], marker='^', label='-', color='y')
    fig=plt.figure()
    ax=Axes3D(fig)
    data=creat_data(100)
    plot_samples(ax,data)
    ax.legend(loc='best')
    plt.show()

def perceptron(train_data,eta,w_0,b_0):
    x=train_data[:,:-1]
    y=train_data[:,-1]
    length=train_data.shape[0]
    w=w_0
    b=b_0
    step_num=0
    while True:
        i=0
        while(i<length):
            step_num+=1
            x_i=x[i].reshape((x.shape[1],1))
            y_i=y[i]
            if y_i*(np.dot(np.transpose(w),x_i)+b)<=0:
                w=w+eta*y_i*x_i
                b=b+eta*y_i
                break
            else:
                i=i+1
        if(i==length):
            break
    return (w,b,step_num)
def creat_hyperplane(x,y,w,b):
    return (-w[0][0]*x-w[1][0]*y-b)/w[2][0]

data=creat_data(100)
eta,w_0,b_0=0.1,np.ones((3,1),dtype=float),1
w,b,num=perceptron(data,eta,w_0,b_0)

fig=plt.figure()
plt.suptitle("perceptron")
ax=Axes3D(fig)
plot_samples(ax,data)
x=np.linspace(-30,100,100)
y=np.linspace(-30,100,100)
x,y=np.meshgrid(x,y)
z=creat_hyperplane(x,y,w,b)
ax.plot_surface(x,y,z,rstride=1,cstride=1,color='g',alpha=0.2)
ax.legend(loc='best')
plt.show()

'''
def creat_data(n):
    np.random.seed(1)
    x_11=np.random.randint(0,100,(n,1))
    x_12=np.random.randint(0,100,(n,1))
    x_13=20+np.random.randint(0,10,(n,1))
    x_21=np.random.randint(0,100,(n,1))
    x_22=np.random.randint(0,100,(n,1))
    x_23=10-np.random.randint(0,10,(n,1))

    new_x_12=x_12*np.sqrt(2)/2-x_13*np.sqrt(2)/2
    new_x_13=x_12*np.sqrt(2)/2+x_13*np.sqrt(2)/2
    new_x_22=x_22*np.sqrt(2)/2-x_23*np.sqrt(2)/2
    new_x_23=x_22*np.sqrt(2)/2+x_23*np.sqrt(2)/2

    plus_samples=np.hstack([x_11,new_x_12,new_x_13,np.ones((n,1))])
    minus_samples=np.hstack([x_21,new_x_22,new_x_23,-np.ones((n,1))])
    samples=np.vstack([plus_samples,minus_samples])
    np.random.shuffle(samples)
    return samples
def plot_samples(ax,samples):
    Y=samples[:,-1]
    Y=samples[:,-1]
    position_p=Y==1
    position_m=Y==-1
    ax.scatter(samples[position_p,0],samples[position_p,1],
        samples[position_p,2],marker='+',label='+',color='b')
    ax.scatter(samples[position_m,0],samples[position_m,1],
        samples[position_m,2],marker='^',label='-',color='y')

def perceptron(train_data,eta,w_0,b_0):
    x=train_data[:,:-1]
    y=train_data[:,-1]
    w=w_0
    b=b_0
    step_num=0
    length=train_data.shape[0]
    while True:
        i=0
        while(i<length):
            step_num+=1
            x_i=x[i].reshape((x.shape[1],1))
            y_i=y[i]
            if y_i*(np.dot(np.transpose(w),x_i)+b)<=0:
                w=w+eta*y_i*x_i
                b=b+eta*y_i
                break
            else:
                i=i+1
        if(i==length ) :
            break
    return (w,b,step_num)

def creat_hyperplane(x,y,w,b):
    return (-w[0][0]*x-w[1][0]*y-b)/w[2][0]
'''
data=creat_data(100)
eta,w_0,b_0=0.1,np.ones((3,1),dtype=float),1
w,b,num=perceptron(data,eta,w_0,b_0)

fig=plt.figure()
plt.suptitle('perceptron')
ax=Axes3D(fig)


plot_samples(ax,data)


x=np.linspace(-30,100,100)
y=np.linspace(-30,100,100)
x,y=np.meshgrid(x,y)
z=creat_hyperplane(x,y,w,b)
ax.plot_surface(x,y,z,rstride=1,cstride=1,color='g',alpha=0.2)

ax.legend(loc='best')
plt.show()


def perceptron_dual(train_data,eta,alpha_0,b_0):
    x=train_data[:,:-1]
    y=train_data[:,-1]
    length=train_data.shape[0]
    alpha=alpha_0
    b=b_0
    step_num=0
    while True:
        i=0
        while i<length:
            step_num+=1
            x_i=x[i].reshape((x.shape[1],1))
            y_i=y[i]
            w=creat_w(train_data,alpha)
            if y_i*(np.dot(np.transpose(w),x_i)+b) <= 0:
                alpha[i]+=eta
                b=b+eta*y_i
                break
            else:
                i=i+1
        if i==length:
            break
    return (alpha,b,step_num)
    
    
    
fig=plt.figure()
plt.suptitle('perceptron')
ax=Axes3D(fig)
plot_samples(ax,data)
x=np.linspace(-30,100,100)
y=np.linspace(-30,100,100)
x,y=np.meshgrid(x,y)
#z=creat_hyperplane(x,y,w,b)
z_2=creat_hyperplane(x,y,w_2,b_2)
#ax.plot_surface(x,y,z,rstride=1,cstride=1,color='g',alpha=0.2)
ax.plot_surface(x,y,z_2,rstride=1,cstride=1,color='c',alpha=0.2)
ax.legend(loc='best')
plt.show()

'''


def creat_w(train_data,alpha):
    print('8')
    x=train_data[:,:-1]
    y=train_data[:,-1]
    N=train_data.shape[0]
    w=np.zeros((x.shape[1],1))
    for i in range(0,N):
        w+=alpha[i][0]*y[i]*(x[i].reshape(x[i].size,1))
    return w

def perceptron_dual(train_data,eta,alpha_0,b_0):
    x=train_data[:,:-1]
    y=train_data[:,-1]
    length=train_data.shape[0]
    alpha=alpha_0
    b=b_0
    step_num=0
    print('6')
    while True:
        i=0
        while(i<length):
            step_num+=1
            x_i=x[i].reshape((x.shape[1],1))
            y_i=y[i]
            w=creat_w(train_data,alpha)
            z=y_i*(np.dot(np.transpose(w),x_i)+b)
            if z<=0:
                alpha[i][0]+=eta
                b+=eta*y_i
                break
            else:
                i+=1
        if(i==length):
            break
    print('7')
    return (alpha,b,step_num)



print('2')

data=creat_data(10)
print('3')
#eta,w_0,b_0=0.1,np.ones((3,1),dtype=float),1
#w,b,num=perceptron(data,eta,w_0,b_0)
alpha,b_2,num_2=perceptron_dual(data,eta=0.1,alpha_0=np.zeros((data.shape[0]*2,1)),b_0=0)
print('4')
w_2=creat_w(data,alpha)
print('5')
#print("w_1,b_1",w,b)
print("w_2,b_2",w_2,b_2)
print('1')

fig=plt.figure()
plt.suptitle('perceptron')
ax=Axes3D(fig)
plot_samples(ax,data)
x=np.linspace(-30,100,100)
y=np.linspace(-30,100,100)
x,y=np.meshgrid(x,y)
#z=creat_hyperplane(x,y,w,b)
z_2=creat_hyperplane(x,y,w_2,b_2)
#ax.plot_surface(x,y,z,rstride=1,cstride=1,color='g',alpha=0.2)
ax.plot_surface(x,y,z_2,rstride=1,cstride=1,color='c',alpha=0.2)
ax.legend(loc='best')
plt.show()











