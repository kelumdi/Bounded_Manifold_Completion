# -*- coding: utf-8 -*-
"""
*   Implementation of the dimensionality reduction method, named as BMC, presented in,
    [1] Gajamannage, Kelum, and Randy Paffenroth. "Bounded manifold completion." 
        Pattern Recognition 111 (2021): 107661.

*   Consider that this implantation demonstrate the first example, a semi-cylinder 
    having a hollow region,  presented in Section 3.1 of [1].
    
*   Please consider that this script might contain cording bug.

*   Please kindly cite [1] if you use BMC for your research.
    
Created on Fri Jul 16 18:37:04 2021.
@author: Kelum Gajamannage.

"""

import numpy as np
import numpy.linalg as la
import pywt

def main():  
      
    rhorZeta = .05 # rho supper script zeta is the panalty parameter of the first constraint, see Eqn.(13) in [1]. 
    rhoEta = .05 # rho supper script eta is the panalty parameter of the second constraint, see Eqn.(13) in [1]. 
    rho = 1.01 # common multiplier for both rhoZeta and rhoEta, see Eqn.(32) in [1].
    
    T = 200 # total number of iteration. 
          
    tol = 10**(-6) # error rolerance, that is the code stops running if rateOfChng < tol, where rateOfChng = frobeniusNorm(L_n-L_(n-1))/frobeniusNorm(M)
    
    r = 4 # number of singular value to be truncated, see Eqn.(4) in [1]. 
    tau = .2 # intializations for K, denoted as K0, and L, denoted as L0, are defined as a linear combination of Dl and Du with respect to the parameter tau. 
     
    Du = np.genfromtxt('Du2.csv', delimiter=',') # squared lower bound of the true squared distance matrix.
    Dl = np.genfromtxt('Dl2.csv', delimiter=',') # squared upper bound of the true squared distance matrix.
         
    n = len(Du)   
             
    # variable intialization
    L0 = tau*Du+(1-tau)*Dl;     K0 = tau*Du+(1-tau)*Dl   
    zeta0 = np.zeros((n,n)); eta0 = np.zeros((n,n))
      
    # callinf the main function
    L = BMC(L0, K0, Du, Dl, zeta0, eta0, r, rhorZeta, rhoEta, rho, T, tol)  
            
    # saving the squared lowrank distance matrix, denoted as K,where Dl < K < Du. 
    np.savetxt('D2.csv',L,delimiter=",")
        
           
def BMC(L, K, Du, Dl, zeta, eta, r, rhorZeta, rhoEta, rho, T, tol):    
    #initial variables
    it = 0; objVal = np.nan;
         
    trunIt = 1 # truncation is perfomed after each trunIt iterations. This is not an essential parameter; but, can change is required. 
    M = (Dl + Du)/2 
    
    print('\n\n----------------------------------------------------------------------------------------')
    print('####  Start: Header labels ####')
    print('Let, M = (Dl + Du)/2')
    print('minOfObj: mininum(nuclearNorm(L) - transpose(Ur*K*transpose(Vr))), Eqn. (10) of [1]')       
    print('rateOfChng: frobeniusNorm(L_n-L_(n-1))/frobeniusNorm(M)')
    print('errOfCons 1: frobeniusNorm(L_n-K_n)/frobeniusNorm(M)')    
    print('errOfCons 2: frobeniusNorm(E_n)/frobeniusNorm(M)') 
    print('rhoZeta: value of the parameter rhoZeta') 
    print('rhoEta: value of the parameter rhoEta') 
    print('####  End: Header labels ####')
    print('\n----------------------------------------------------------------------------------------')
    print('IterNum\t minOfObj \t rateOfChng L \t errOfCons 1 \t errOfCons 2 \t rhoZeta \t rhoEta\n')
        
    Ur,Vr = rSV(K,r) # computing Al and Bl for the  first time
   
    while True:
        # Break if we use too many interations               
        it += 1
        if it > T:
            print('\nReached maximum iterations')           
            break          
        
        # update L
        GN = K+1/rhorZeta*zeta  
        
        LN = gramShrinkage(GN, rhorZeta, T, it)               
              
        if (it%trunIt)==0:
            Ur,Vr = rSV(K,r)          

        # update L, new updates: KN     
        KN = 1/rhorZeta*(rhorZeta*LN + (np.matmul(Ur.T,Vr)+zeta) - eta)   
        
        # update zetabda, new updates: KN, LN, EN, L
        zetaN = zeta + rhorZeta*(LN-KN)       
               
        # E shrinkage of LN      
        EN = Eshrinkage(KN,Du,Dl)
        
        # update etama, new updates: KN, LN, zetaN, EN
        etaN = eta + rhoEta*EN
               
        # objective value    
        objVal = np.linalg.norm(LN,'nuc') - np.trace(np.matmul(Ur,np.matmul(KN,Vr.T)))
              
        # errors 
        errL  = la.norm(LN-L, ord='fro')/la.norm(M, ord='fro') 
        errKL = la.norm(LN-KN,ord='fro')/la.norm(M, ord='fro')
        errE = la.norm(EN,ord='fro')/la.norm(M, ord='fro')        
           
        if it==1: # display statistics            
            print('%.0f\t %.4e\t %.4e\t %.4e\t %.4e\t %.4e\t %.4e' % (it,objVal,errL,errKL,errE,rhorZeta,rhoEta))
            
        if (it%10)==0: # display statistics            
            print('%.0f\t %.4e\t %.4e\t %.4e\t %.4e\t %.4e\t %.4e' % (it,objVal,errL,errKL,errE,rhorZeta,rhoEta))
#            print('%.4e\t %.4e\t %.4e\t' % (rhorZeta,rho_mu,rhoEta))
            
        if ~np.isnan(errL):           
            if errL < tol:
                print('%.0f\t %.4e\t %.4e\t %.4e\t %.4e\t %.4e\t %.4e' % (it,objVal,errL,errKL,errE,rhorZeta,rhoEta))
                print('\nReached tolerance')
                break          
        
        # updating parameters
        rhorZeta = rho*rhorZeta; rhoEta = rho*rhoEta           
  
        # moving to the new iteration
        L = LN; K = KN;   zeta = zetaN; eta = etaN    
    
    return L  

def gramShrinkage(G, rhorZeta, T, it):  # checked for errors
    n = len(G)
    
    # converting G to Gramian matrix
    offDiagG = G - np.diag(np.diag(G))
    J = np.identity(n) - np.ones((n,n))/n
    gram = -.5*np.matmul(J,np.matmul(offDiagG,J))  
    
    s, U = np.linalg.eigh(gram)
    s = np.real(s); U = np.real(U)  
    
    if np.min(s)<0:        
        s = s + np.min(s) - 1/rhorZeta
    else:
        s = s - 1/rhorZeta
    
    s_shrink = pywt.threshold(s,0, 'greater')

    sN = np.diag(s_shrink)     
    gramN = np.matmul(U,np.matmul(sN,U.T))

    # converting Gramian to G
    Ge = np.matmul(np.diag(gramN).reshape(n,1),np.ones((1,n))) \
    + np.matmul(np.ones((n,1)),np.diag(gramN).reshape(1,n))
    GN = Ge - 2*gramN
    
    return GN

def Eshrinkage(K,Du,Dl):
    n = len(K)
    E = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if K[i,j] > Du[i,j]:
                E[i,j] = K[i,j] - Du[i,j]
            elif K[i,j] < Dl[i,j]:
                E[i,j] = K[i,j] - Dl[i,j]
            else:
                E[i,j] = 0  
  
    return E

def rSV(K,r): 
    # computing Ur and Vr
    U, S, V = np.linalg.svd(K)
    V = V.T
    Ur = U[:,0:r].T
    Vr = V[:,0:r].T
    return Ur, Vr

main()