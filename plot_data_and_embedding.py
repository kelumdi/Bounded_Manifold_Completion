# -*- coding: utf-8 -*-
"""
*   Plot the dataset and the 2D-embedding of the first example, a semi-cylinder having a hollow region,  
    presented in Section 3.1 of [1].
    [1] Gajamannage, Kelum, and Randy Paffenroth. "Bounded manifold completion." 
        Pattern Recognition 111 (2021): 107661.
        
*   Please consider that this script might contain cording bugs.

*   Please kindly cite [1] if you use BMC for your research.
        
Created on Fri Jul 16 18:37:04 2021.
@author: kgajamannage.

"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

plt.close('all')

dt = np.genfromtxt('dt.csv', delimiter=',')

########  plotting datadata
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.scatter(dt[:,0],dt[:,1],dt[:,2], c = 'b', marker='o')
plt.title('Data')

######## computing and plotting the embedding
D = np.genfromtxt('D2.csv', delimiter=',') # squared distance matrix
n = len(D)

# first, use the following steps to get embedding from the distance matrix.
# the next three lines convert the squared distance matrix to its gramian
offDiagDg = D - np.diag(np.diag(D)) 
J = np.identity(n) - np.ones((n,n))/n
grm = -.5*np.matmul(J,np.matmul(offDiagDg,J))

[u,Sgrm,vt] = la.svd(grm) # SVD of the gramian
Sgrm = np.sqrt(Sgrm) # square root of the gramian
emb = np.matmul(u[:,0:2],np.diag(Sgrm[0:2])) # 2D embedding

# second, plot the embedding
ax = fig.add_subplot(122)
ax.scatter(emb[:,0],emb[:,1], c = 'b', marker='o')
plt.title('Embedding')

