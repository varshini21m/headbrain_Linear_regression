#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']= (20.0,10.0)

# reading data
data = pd.read_csv(r'C:\Users\HP\Downloads\headbrain.csv')
print(data.shape)
data.head()


# In[2]:


# collecting x and y
X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values


# In[3]:


#mean of x and y
mean_x = np.mean(X)
mean_y = np.mean(Y)

# total number of values
n = len(X)

# using the formula to cal b1:is mean and b0:is C intercept 
# m = sum((x-x')(y-y'))
#     __________________
#       sum((x-x')^2)

numer = 0
denom = 0
for i in range(n):
    numer += (X[i] - mean_x)*(Y[i] - mean_y)
    denom += (X[i] - mean_x)**2
b1 = numer / denom
b0 = mean_y - (b1 * mean_x)
# y = mx + c now here became b0 = b1x + c

print(b1,b0)


# In[5]:


# plotting values and regression line

max_x = np.max(X) + 100
min_x = np.min(X) - 100

# calculating line values x and y
x = np.linspace(min_x, max_x, 1000)
y = b0 + b1 * x

# plotting line
plt.plot(x,y,color='#58b970', label='Regression Line')
# plotting scatter points
plt.scatter(X,Y,c='#ef5423', label='Scatter PLot')

plt.xlabel('head size in cm^3')
plt.ylabel('brain weight in grams')
plt.legend()
plt.show()


# In[6]:


# R^2 method to check the goodness of fit
ss_t = 0
ss_r = 0
for i in range(n):
    y_pred = b0 +b1 * X[i] # y = mx + c
    ss_t += (Y[i] - mean_y)**2
    ss_r += (Y[i] - y_pred)**2
r2 =  1- (ss_r/ss_t)
print(r2)


# In[7]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# cannot use rank 1 matrix in scikit learn
X = X.reshape((n,1))
# creating model
reg = LinearRegression()
# fitting training data
reg = reg.fit(X,Y)
# y prediction
Y_pred = reg.predict(X)

# calculating r2 score
r2_score = reg.score(X,Y)

print(r2_score)


# In[ ]:





# In[ ]:




