#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Create list
l=[1,2,3,4]
print(l)


# In[ ]:


l=[[1,'c'],[3,4.5]]
print(l)


# In[ ]:


l=[1,2,3,4]
l=l+1


# In[ ]:


import numpy

# Change a list into an array
a=numpy.array([1,2,3,4])
print(a)


# In[ ]:


a=a+1
print(a)


# In[ ]:


# Create a matrix
b=numpy.array([[1,2,3,4],[3,2,4,2],[4,8,7,1]])
print(b)


# In[ ]:


# Shape of a matrix
print(b.shape)


# In[ ]:


print(b.shape[0])


# In[ ]:


# Access the matrix element
# first line, first colums
print(b[0,0])


# In[ ]:


# Second line, third column
print(b[1][2])


# In[ ]:


# Second line, third column
print(b[1,2])


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
print(b.T)


# In[ ]:


# Square matrix
b=b[0:3,0:3]
print(b)


# In[ ]:


# Symmetric matrix
print(b+b.T)


# In[ ]:


# Antisymmetric matrix
print(b-b.T)


# In[ ]:


# Inverse matrix
ib=numpy.linalg.inv(b)
print(ib)


# In[ ]:


# Element by element
print(b*ib)


# In[ ]:


# Matrix product
print(numpy.dot(b,ib))


# In[ ]:


# Determinant
print(numpy.linalg.det(b))


# In[ ]:


# Identity matrices
print(numpy.eye(3))


# In[ ]:


# Null matrices
print(numpy.zeros((3,4)))


# In[ ]:


# Constant matrix
print(numpy.full((2,3), fill_value=0.5))


# In[ ]:


# Eigenvalues and eigenvectors
b=b+b.T
evalue, evect = numpy.linalg.eig(b)
print("Eigenvalues")
print(evalue)
print("Eigenvectors")
print(evect)


# In[ ]:


# First eigenvector (First column of evect matrix)
print(evect[:,0])


# In[ ]:


# Be careful when copying!! 
# ¡¡¡This is wrong!!!

c=b
c[0,0]=-1
print("c")
print(c)
print("b")
print(b)


# In[ ]:


# Be careful when copying!! 
# ¡¡¡This is right!!!

import copy
c=copy.deepcopy(b)

c[0,0]=-2
print("c")
print(c)
print("b")
print(b)


# In[ ]:


# Alternative

from copy import deepcopy
c=deepcopy(b)

