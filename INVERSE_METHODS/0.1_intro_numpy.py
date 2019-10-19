#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


# Arrays and matrices

import numpy as np

# Create list
l=[1,2,3,4]
print(l)

l=[[1,'c'],[3,4.5]]
print(l)

l=l+1


# In[ ]:


# Change a list into an array
a=np.array([1,2,3,4])
print(a)
a=a+1
print(a)


# In[ ]:


# Create a matrix
b=np.array([[1,2,3,4],[3,2,4,2],[4,8,7,1]])
print(b)


# In[ ]:


# Shape of a matrix
print(b.shape[1])


# In[ ]:


# Access the matrix element
# first line, first colums
print(b[0,0])


# In[ ]:


# Second line, third column
print(b[1][2])


# In[ ]:


# First line
print(b[0])


# In[ ]:


# First line (alternative)
print(b[0,:])


# In[ ]:


# Second column
print(b[:,1])


# In[ ]:


# Submatrix
print(b[0:2,0:2])


# In[ ]:


# Submatrix
print(b[::2,::2])


# In[ ]:


# Transpose
b=b[0:3,0:3]
print(b+b.T)
print(b-b.T)


# In[ ]:


# Inverse matrix
ib=np.linalg.inv(b)
print(ib)


# In[ ]:


# Element by element
print(b*ib)


# In[ ]:


# Matrix product
print(np.dot(b,ib))


# In[ ]:


# Determinant
print(np.linalg.det(b))


# In[ ]:


# Eigenvalues and eigenvectors
evalue, evect = np.linalg.eig(b)
print("Eigenvalues=",evalue)
print("Eigenvectors=",evect)


# In[ ]:


# First eigenvector (First column of evect matrix)
print(evect[:,0])


# In[ ]:


# Be careful when copying!! 
# ¡¡¡This is wrong!!!

c=b
c[0,0]=-1
print(c)
print(b)


# In[ ]:


# Be careful when copying!! 
# ¡¡¡This is right!!!

import copy
c=copy.deepcopy(b)

# Alternative
#from copy import deepcopy
#c=deepcopy(b)

c[0,0]=-2
print(c)
print(b)


# In[ ]:




