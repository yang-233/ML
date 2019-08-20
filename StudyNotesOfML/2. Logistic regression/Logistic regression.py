#!/usr/bin/env python
# coding: utf-8

# 
# # 读入数据

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


path = 'ex2data1.txt'
exam1 = 'exam1'
exam2 = 'exam2'
admitted = 'admitted'
data = pd.read_csv(path, header=None, names=[exam1, exam2, admitted])
# data.head()


# # 可视化

# In[3]:


positive = data[data[admitted].isin([1])]
negative = data[data[admitted].isin([0])]

# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(positive[exam1], positive[exam2], s=50, c='b', marker='o', label='Admitted')
# ax.scatter(negative[exam1], negative[exam2], s=50, c='r', marker='x', label='Not Admitted')
# ax.legend()
# ax.set_xlabel('Exam 1 Score')
# ax.set_ylabel('Exam 2 Score')


# # sigmod函数

# In[4]:


def sigmod(z):
    return 1. / (1. + np.exp(-z))


# In[6]:


# nums = np.arange(-20, 20)
# fig, ax = plt.subplots(figsize=(12, 8))
# ax.plot(nums, sigmod(nums), 'r')


# # 损失函数

# In[42]:


def cost_func(X, y, theta, m, EPS=0):
    h = sigmod(X * theta)
    #print(h)
    # first = np.log(h)
    # second = np.log(1 - h)

    return -np.sum(np.multiply(y, np.log(h + EPS)) + np.multiply(1 - y, np.log(1 - h + EPS))) / m


# # 初始化输入输出

# In[32]:


rows = data.shape[0]
cols = data.shape[1]
# rows, cols


# In[12]:


X = np.mat(np.ones((rows, cols)))
X[:, 1:] = data.iloc[:, 0:cols-1].values
# X[:5,:]


# In[20]:


y = np.mat(data.iloc[:,cols-1].values).T
# y[:5,:]


# In[59]:


theta = np.mat([0., 0., 0.], dtype='float64').T
# theta


# In[29]:


# X.shape, theta.shape, y.shape


# In[61]:


# cost_func(X, y, theta, rows)


# # 梯度下降

# In[62]:


#O(iters * n * m * n * n) 
def batch_gradient_decent(X, y, theta, m, alpha=0.01, num_of_iters=1000):
    #获取参数数量
    num_of_parameters = theta.shape[0]
    #保存损失函数值
    cost_list = []
    #用于保存theta的临时向量
    theta_tmp = theta.copy()
    for i in range(num_of_iters):
        bias = sigmod(X * theta) - y
        for j in range(num_of_parameters):
            theta_tmp[j, 0] = theta[j, 0] - (alpha / m) * np.sum(np.multiply(bias, X[:, j]))
        theta = theta_tmp
        cost_list.append(cost_func(X, y, theta, rows))
    return theta, cost_list


# In[64]:


theta, cost_values = batch_gradient_decent(X, y, theta, rows, 0.0007, 2000)
print(cost_values[-1])
# len(cost_values)


# In[ ]:





# In[ ]:




