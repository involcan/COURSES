#!/usr/bin/env python
# coding: utf-8

# Create a list
l=[1,2,3,4]
print(l)


l=[[1,'c'],[3,4.5]]
print(l)


l=[1,2,3,4]
l=l+1


import numpy

# Change a list into an array
a=numpy.array([1,2,3,4])
print(a)


a=a+1
print(a)


# Create a matrix
b=numpy.array([[1,2,3,4],[3,2,4,2],[4,8,7,1]])
print(b)


# Shape of a matrix
print(b.shape)


print(b.shape[0])


# Access the matrix element
# first line, first columns
print(b[0,0])


# Second line, third column
print(b[1][2])


# Second line, third column
print(b[1,2])


# First line
print(b[0])


# First line (alternative)
print(b[0,:])


# Second column
print(b[:,1])


# Submatrix
print(b[0:2,0:2])


# Submatrix
print(b[::2,::2])


# In[ ]:


# Transpose
print(b.T)


# Square matrix
b=b[0:3,0:3]
print(b)


# Symmetric matrix
print(b+b.T)


# Antisymmetric matrix
print(b-b.T)


# Inverse matrix
ib=numpy.linalg.inv(b)
print(ib)


# Element by element
print(b*ib)


# Matrix product
print(numpy.dot(b,ib))


# Determinant
print(numpy.linalg.det(b))


# Identity matrices
print(numpy.eye(3))


# Null matrices
print(numpy.zeros((3,4)))


# Constant matrix
print(numpy.full((2,3), fill_value=0.5))


# Eigenvalues and eigenvectors
b=b+b.T
evalue, evect = numpy.linalg.eig(b)
print("Eigenvalues")
print(evalue)
print("Eigenvectors")
print(evect)


# First eigenvector (First column of evect matrix)
print(evect[:,0])


# Be careful when copying!! 
# ¡¡¡This is wrong!!!

c=b
c[0,0]=-1
print("c")
print(c)
print("b")
print(b)


# Be careful when copying!! 
# ¡¡¡This is right!!!

import copy
c=copy.deepcopy(b)

c[0,0]=-2
print("c")
print(c)
print("b")
print(b)


# Alternative

from copy import deepcopy
c=deepcopy(b)

