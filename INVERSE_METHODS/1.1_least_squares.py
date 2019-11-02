#!/usr/bin/env python
# coding: utf-8

# # 1.1 Least squares
# ## by Luca D'Auria (ldauria@iter.es)
# ## Instituto Volcanológico de Canarias (INVOLCAN)
# ## www.involcan.org

# You need to run this cell only if using the online nbviewer
import sys
get_ipython().system(u'conda install --yes --prefix {sys.prefix} numpy matplotlib')


# ### Straight line fit
# 
# *Forward problem*:
# $y_i=a x_i + b$
# 
# *True model*:
# $a=2; b=-3$
# 
# #### 1) Data

import numpy as np
import matplotlib.pyplot as plt

x=np.array([1,2,3,4])

# Number of data
N=len(x)
# Number of parameters
M=2

# Generate data with noise
a=2
b=-3
y = a*x + b + np.random.randn(N)*0.5

plt.plot(x,y,'ko')
plt.xlim([0,5])
plt.show()


# #### 2) Kernel


G=np.empty((N,M))

for i in range(N):
    G[i,0]=x[i]  # a
    G[i,1]=1     # b
    
print(G)


# #### 3) Inverse operator
# $G^{-1}=\left( G^{T} G \right)^{-1} G^{T}$


G1=np.dot(G.T,G)
G2=np.linalg.inv(G1)
Gi=np.dot(G2,G.T)

# or alternatively...
Gi=np.dot(np.linalg.inv(np.dot(G.T, G)), G.T)

print(Gi)


# #### 4) Least squares solution


m=np.dot(Gi,y.T)
print(m)


# #### 5) Checking the result


# Forward modeling
ysynth=np.dot(G,m)

# Observed data
plt.plot(x,y,'ko')
# Synthetic data
plt.plot(x,ysynth,'r-')
# Residuals
for i in range(N):
    plt.plot([x[i],x[i]],[y[i],ysynth[i]],'b-')
    
plt.xlim([0,5])
plt.show()


# ### Polynomial fit
# 
# *Forward problem*:
# $y_i=a x_i^3 + b x_i^2 + c x_i + d$
# 
# *True model*:
# $a=0.2; b=-1; c=-1; d=2$
# 
# #### 1) Data


x=np.array([1,1.5,1.8,2,3,3.5,4,4.2,4.7,5])

# Number of data
N=len(x)
# Number of parameters
M=4

# Generate data with noise
a=0.2
b=-1
c=-1
d=2
y = a*x**3 + b*x**2 + c*x + d + np.random.randn(N)*0.2

plt.plot(x,y,'ko')
plt.xlim([0,6])
plt.show()


# #### 2) Kernel


G=np.empty((N,M))

for i in range(N):
    G[i,0]=x[i]**3  # a
    G[i,1]=x[i]**2  # b
    G[i,2]=x[i]     # c
    G[i,3]=1        # d
    
print(G)


# #### 3) Inverse operator
# 
# $G^{-1}=\left( G^{T} G \right)^{-1} G^{T}$


Gi=np.dot(np.linalg.inv(np.dot(G.T, G)), G.T)


# #### 4) Least squares solution


m=np.dot(Gi,y.T)
print(m)


# #### 5) Checking the result


# Forward modeling
ysynth=np.dot(G,m)

# Observed data
plt.plot(x,y,'ko')
# Synthetic data
plt.plot(x,ysynth,'r-')
# Residuals
for i in range(N):
    plt.plot([x[i],x[i]],[y[i],ysynth[i]],'b-')
    
plt.xlim([0,6])
plt.show()


# ### Plane fit
# 
# *Forward problem*:
# $z_i=a x_i + b y_i + c$
# 
# *True model*:
# $a=1; b=2; c=-3$
# 
# #### 1) Data


x=np.array([1,2,3,4,2,3,4])
y=np.array([3,2,1,3,4,1,4])

# Number of data
N=len(x)
# Number of parameters
M=3

# Generate data with noise
a=1
b=2
c=-3
z = a*x + b*y + c + np.random.randn(N)

# Basic 3D graphics with Matplotlib
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z, 'ko')
plt.show()


# #### 2) Kernel


G=np.empty((N,M))

for i in range(N):
    G[i,0]=x[i]  # a
    G[i,1]=y[i]  # b
    G[i,2]=1     # c
    
print(G)


# #### 3) Inverse operator
# 
# $G^{-1}=\left( G^{T} G \right)^{-1} G^{T}$


Gi=np.dot(np.linalg.inv(np.dot(G.T, G)), G.T)


# #### 4) Least squares solution


m=np.dot(Gi,z.T)
print(m)


# #### 5) Checking the result


# Forward modeling
zsynth=np.dot(G,m)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plane with synthetic data
Xg, Yg = np.meshgrid(np.arange(0, 6, 0.5), np.arange(0, 6, 0.5))
Zg = m[0]*Xg + m[1]*Yg + m[2]
ax.plot_surface(Xg, Yg, Zg, linewidth=0, alpha=0.5)

# Data
ax.scatter(x, y, zsynth, 'ko')

# Residuals
for i in range(N):
    plt.plot([x[i],x[i]],[y[i],y[i]],[z[i],zsynth[i]],'b-')

plt.show()

