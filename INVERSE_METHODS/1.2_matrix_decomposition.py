#!/usr/bin/env python
# coding: utf-8

# # 1.2 Matrix decomposition
# ## by Luca D'Auria (ldauria@iter.es)
# ## Instituto Volcanológico de Canarias (INVOLCAN)
# ## www.involcan.org

# You need to run this cell only if using the online nbviewer
import sys
get_ipython().system(u'conda install --yes --prefix {sys.prefix} numpy matplotlib')


# ### Diagonal matrices


import numpy as np

dm = np.diag([5,3,2,1])
print(dm)



m=np.array([[2,3,1],[2,3,0],[4,-1,-1]])
print(m)
print("Diag =",np.diag(m))


# ### Orthogonal matrices


Mo=np.array([[2,6,3],[3,2,-6],[6,-3,2]])/7

print(Mo)


# #### $M_1 = M_o M_o^T$
# #### $M_2 = M_o^T M_o$


M1=np.dot(Mo,Mo.T)
print("M1")
print(np.round(M1))

M2=np.dot(Mo,Mo.T)
print("M2")
print(np.round(M2))


# ## Matrix as linear operators


import matplotlib.pyplot as plt

M=np.array([[1,1],[0,2]])
print("M")
print(M)

# Original vector
v=np.array([0.5,1])

# Transformed vectors
vt=np.dot(M,v.T)
print("vt=",vt)

# Plot
plt.arrow(0,0,v[0],v[1], length_includes_head=True, head_width=0.05, fc='k', ec='k')
plt.arrow(v[0],v[1],vt[0],vt[1], length_includes_head=True, head_width=0.05, fc='b', ec='b')

# Equal aspect ratio
plt.xlim([0,3])
plt.ylim([0,3])
plt.gca().set_aspect('equal','box')

plt.show()



# Original vectors
ang=np.linspace(0,2*np.pi,37)
x=np.cos(ang)
y=np.sin(ang)
v=np.vstack((x,y))

# Transformed vectors
vt=np.dot(M,v)

for i in range(len(ang)):
    # Plot original vectors
    plt.plot([0,v[0,i]],[0,v[1,i]],'k')
    # Plotting transformed vector
    plt.plot([v[0,i],v[0,i]+vt[0,i]],[v[1,i],v[1,i]+vt[1,i]],'b')

# Equal aspect ratio
plt.gca().set_aspect('equal', 'box')

plt.show()


# ### Pure stretching


M=np.diag([2,0.7])

print("M")
print(M)

# Original vectors
ang=np.linspace(0,2*np.pi,37)
x=np.cos(ang)
y=np.sin(ang)
v=np.vstack((x,y))

# Transformed vectors
vt=np.dot(M,v)

# Plot vectors
for i in range(len(ang)):
    plt.plot([0,v[0,i]],[0,v[1,i]],'k')
    plt.plot([v[0,i],v[0,i]+vt[0,i]],[v[1,i],v[1,i]+vt[1,i]],'b')

# Equal aspect ratio
plt.gca().set_aspect('equal', 'box')

plt.show()


# ### Pure rotation


# Rotation of 40º counterclockwise
ang=40*np.pi/180

M=np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])

print("M")
print(M)

# Original vectors
ang=np.linspace(0,2*np.pi,37)
x=np.cos(ang)
y=np.sin(ang)
v=np.vstack((x,y))

# Transformed vectors
vt=np.dot(M,v)

# Plot vectors
for i in range(len(ang)):
    plt.plot([0,v[0,i]],[0,v[1,i]],'k')
    plt.plot([v[0,i],v[0,i]+vt[0,i]],[v[1,i],v[1,i]+vt[1,i]],'b')

plt.gca().set_aspect('equal', 'box')

plt.show()


# ## Eigenvalues and eigenvectors


M=np.array([[1,1],[0,2]])

print("M")
print(M)

evalue, evect = np.linalg.eig(M)

evect1=evect[:,0]
evect2=evect[:,1]

print("evalue=",evalue)
print("evect 1 =",evect1)
print("evect 2 =",evect2)



# Are eigenvector normalized?
print(np.linalg.norm(evect1))
print(np.linalg.norm(evect2))



# Transformed vectors
vt=np.dot(M,v)

# Transformed eigenvectors
evect1t=np.dot(M,evect1.T)
evect2t=np.dot(M,evect2.T)

# Plot vectors
for i in range(len(ang)):
    plt.plot([0,v[0,i]],[0,v[1,i]],'k')
    plt.plot([v[0,i],v[0,i]+vt[0,i]],[v[1,i],v[1,i]+vt[1,i]],'b')

# Plotting eigenvectors
plt.plot([0,evect1[0],evect1[0]+evect1t[0]],[0,evect1[1],evect1[1]+evect1t[1]],'r', linewidth=2)
plt.plot([0,evect2[0],evect2[0]+evect2t[0]],[0,evect2[1],evect2[1]+evect2t[1]],'r', linewidth=2)

plt.gca().set_aspect('equal', 'box')

plt.show()


# ### Uses of eigenvalues
# 
# #### 1) Trace of a matrix $Tr\left(\mathbf{M}\right) = \sum M_{ii} = \sum \lambda_i$


tr1=np.sum(np.diag(M))
tr2=np.sum(evalue)

print("tr1=",tr1," tr2=",tr2)


# #### Determinant of a matrix $det\left( \mathbf{M}\right) = \prod \lambda_i$


det1=np.linalg.det(M)
det2=np.prod(evalue)

print("det1=",det1," det2=",det2)


# ### Eigendecomposition of a matrix
# 
# $\mathbf{M} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^{-1}$


evalue, V = np.linalg.eig(M)
Lambda = np.diag(evalue)

Vi=np.linalg.inv(V)

# Recostruction of M from its eigendecomposition
MM = np.dot(V,np.dot(Lambda,Vi))

print("M")
print(M)
print("MM")
print(MM)


# ### Diagonalizable matrices


# Rotation matrix: an example of non diagonalizable matrix

ang=30*np.pi/180
M=np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])

print("M")
print(M)

evalue, evect = np.linalg.eig(M)

evect1=evect[:,0]
evect2=evect[:,1]

print("evalue=",evalue)
print("evect 1 =",evect1)
print("evect 2 =",evect2)


# ### Using the eigendecomposition for matrix inversion
# $\mathbf{M}^{-1} = \mathbf{V} \mathbf{\Lambda}^{-1} \mathbf{V}^{-1}$


Lambda_i = np.diag(1/evalue)

Mi = np.dot(V,np.dot(Lambda_i,Vi))

# Verify Mi is the inverse of M
print(np.round(np.dot(M,Mi)))


# ## Singular value decomposition
# $\mathbf{M} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$


M=np.array([[2,1],[2,2],[-1,0]])

U, S, VT = np.linalg.svd(M)

print("U ",U.shape)
print("S ",S.shape)
print("VT ",VT.shape)



# Let's check it
Sigma=np.zeros((3,2))
Sigma[:2, :2] = np.diag(S)

print("Sigma")
print(Sigma)

MM=np.dot(U,np.dot(Sigma,VT))

print("M")
print(M)

print("MM")
print(np.round(MM))



# Is U orthogonal?
print(np.round(np.dot(U,U.T)))



# Is V orthogonal?
print(np.round(np.dot(VT,VT.T)))


# ### Pseudoinverse with SVD
# Definition: $\mathbf{M}^{-g} \mathbf{M} = \mathbf{I}$
# 
# Using SVD: $\mathbf{M}^{-g} = \mathbf{V} \mathbf{\Sigma}^{-1} \mathbf{U}^T$


V=VT.T

Sigma_i = np.zeros((2,3))
Sigma_i[:2, :2] = np.diag(1./S)

UT = U.T

Mg = np.dot(V,np.dot(Sigma_i,UT))

print(np.round(np.dot(M,Mg)))

