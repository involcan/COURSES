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
# $a=2; b=1$
# 
# #### 1) Data

import numpy as np
import matplotlib.pyplot as plt

x=np.array([1,2,3,4])

a=2
b=1
y = a*x + b + np.random.random(4)

plt.plot(x,y,'ko')
plt.xlim([0,5])
plt.show()


# #### 2) Kernel

G=np.empty((4,2))

for i in range(4):
    G[i,0]=x[i]  # a
    G[i,1]=1     # b
    
print(G)


# #### 3) Inverse operator
# $G^{-1}=\left( G^{T} G \right)^{-1} G^{T}$

G1=np.dot(G.T,G)
G2=np.linalg.inv(G1)
G3=np.dot(G2,G.T)

# or alternatively...
G3=np.dot(np.linalg.inv(np.dot(G.T, G)), G.T)

print(G3)


# #### 4) Least squares solution

m=np.dot(G3,y.T)
print(m)


# #### 5) Checking the result

# Forward modeling
ysynth=np.dot(G,m)

plt.plot(x,y,'ko')
plt.plot(x,ysynth,'r-')
plt.xlim([0,5])
plt.show()


# ### Quadratic fit
# 
# *Forward problem*:
# $y_i=a x_i^2 + b x_i + c$
# 
# *True model*:
# $a=1; b=-1; c=1$
# 
# #### 1) Data


x=np.array([1,2,3,4,5])

a=1
b=-1
c=1
y = a*x**2 - b*x + c + np.random.random(5)

plt.plot(x,y,'ko')
plt.xlim([0,6])
plt.show()


# #### 2) Kernel

G=np.empty((5,3))

for i in range(5):
    G[i,0]=x[i]**2  # a
    G[i,1]=x[i]     # b
    G[i,2]=1        # c
    
print(G)


# #### 3) Inverse operator
# 
# $G^{-1}=\left( G^{T} G \right)^{-1} G^{T}$

Gi=np.dot(np.linalg.inv(np.dot(G.T, G)), G.T)

print(Gi)


# #### 4) Least squares solution

m=np.dot(Gi,y.T)
print(m)


# #### 5) Checking the result

# Forward modeling
ysynth=np.dot(G,m)

plt.plot(x,y,'ko')
plt.plot(x,ysynth,'r-')
plt.xlim([0,5])
plt.show()


# ### Plane fit
# 
# *Forward problem*:
# $z_i=a x_i + b y_i + c$
# 
# *True model*:
# $a=1; b=2; c=-1$
# 
# #### 1) Data

x=np.array([1,2,3,4,2,3,4])
y=np.array([3,2,1,3,4,1,4])

a=1
b=2
c=-1
z = a*x + b*y + c + np.random.random(7)

# Basic 3D graphics with Matplotlib
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, 'ko')
plt.show()


# #### 2) Kernel

G=np.empty((7,3))

for i in range(7):
    G[i,0]=x[i]  # a
    G[i,1]=y[i]  # b
    G[i,2]=1     # c
    
print(G)


# #### 3) Inverse operator
# 
# $G^{-1}=\left( G^{T} G \right)^{-1} G^{T}$

Gi=np.dot(np.linalg.inv(np.dot(G.T, G)), G.T)

print(Gi)


# #### 4) Least squares solution

m=np.dot(Gi,y.T)
print(m)


# #### 5) Checking the result

# Forward modeling
zsynth=np.dot(G,m)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plane with synth data
Xg, Yg = np.meshgrid(np.arange(0, 5, 0.5), np.arange(0, 5, 0.5))
Zg = m[0]*Xg + m[1]*Yg + m[2]
ax.plot_surface(Xg, Yg, Zg, linewidth=0, alpha=0.5)

ax.scatter(x, y, zsynth, 'ro')

plt.show()

