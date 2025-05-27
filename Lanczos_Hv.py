
import matplotlib.pylab as plt
from termcolor import colored
from collections import deque
import numpy as np
import pickle
import time

class Lanc_Hv():
	'''
 	The onsite optimisation in two-state dmrg algorithm reduces to solving an eigenvlaue problem $Hv = \Lambda v$. This is solved using the Lanczos algorithm.
 	'''
	def Hv(L, R, GV, wi, wj):   
		'''
  		L : Left Env
    		R : Right Env
      		GV : Guess vector
		wi,wj :  MPO operators on current two sites.
  
  		Implements The Lanczos algorithm which is an iterative numerical method used to approximate the eigenvalues and eigenvectors of large Hermitian matrices. This makes it particularly effective for extracting extreme eigenvalues (e.g., ground state energies) in large-scale quantum systems, where full diagonalization is computationally impractical.
  		'''
	
		S = 0
		w1=[]
		
		p = np.tensordot(np.tensordot(L[0], GV[0], axes=(0,0)), R[0], axes=(5,0))      #[vR] wL wR vR*  [vL] d d vR = wL wR vR* d d [vR]  [vL] wL wR vL* = wL wR vR* d d wL wR vL*
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))   # wL [wR] vR* [d] d wL wR vL*  [wL] wR d [d*] = wL vR* [d] [wL] wR vL* [wR] d  [wL] [wR] d [d*] = wL vR* wR vL* d d

		p = np.tensordot(np.tensordot(L[0], GV[4], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[2], GV[3], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[2], GV[2], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[1], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))
	
		S1 = S.copy()
		w1.extend(list(S[0,:,0,:,0,0].flatten()))

		S = 0
	
		p = np.tensordot(np.tensordot(L[0], GV[0], axes=(0,0)), R[1], axes=(5,0))      
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[4], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))
	
		p = np.tensordot(np.tensordot(L[2], GV[2], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[1], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[2], GV[3], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		w1.extend(list(S[0,:,0,:,0,1].flatten()))

		S = 0
	
		p = np.tensordot(np.tensordot(L[1], GV[0], axes=(0,0)), R[0], axes=(5,0))      
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0])) 

		p = np.tensordot(np.tensordot(L[1], GV[4], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))
	
		p = np.tensordot(np.tensordot(L[3], GV[2], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[1], GV[1], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[3], GV[3], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		w1.extend(list(S[0,:,0,:,1,0].flatten()))

		S = 0
	
		p = np.tensordot(np.tensordot(L[1], GV[0], axes=(0,0)), R[1], axes=(5,0))      
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0])) 

		p = np.tensordot(np.tensordot(L[1], GV[4], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))
	
		p = np.tensordot(np.tensordot(L[3], GV[2], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[1], GV[1], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[3], GV[3], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		w1.extend(list(S[0,:,0,:,1,1].flatten()))
		w1.extend(list(S1[0,:,0,:,1,1].flatten()))
		
		return w1

	def Hv_sA(L, R, GV, wi, wj):
		#[B0rr10,B0rr30,B0rr21,B1er10,B1er30,B1er21,B0er40]
		S = 0
		w1=[]
		
		p = np.tensordot(np.tensordot(L[0], GV[0], axes=(0,0)), R[0], axes=(5,0))      #[vR] wL wR vR*  [vL] d d vR = wL wR vR* d d [vR]  [vL] wL wR vL* = wL wR vR* d d wL wR vL*
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))   # wL [wR] vR* [d] d wL wR vL*  [wL] wR d [d*] = wL vR* [d] [wL] wR vL* [wR] d  [wL] [wR] d [d*] = wL vR* wR vL* d d

		p = np.tensordot(np.tensordot(L[0], GV[1], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,2:3], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[6], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,3:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[2], GV[5], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,1:2], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[2], GV[3], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))
	
		p = np.tensordot(np.tensordot(L[2], GV[4], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,2:3], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[2], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,1:2], axes=([2,3,6],[3,1,0]))
	
		S1 = S.copy()
		w1.extend(list(S[0,:,0,:,0,0].flatten()))
		w1.extend(list(S[0,:,0,:,0,2].flatten()))

		S = 0
	
		p = np.tensordot(np.tensordot(L[0], GV[0], axes=(0,0)), R[1], axes=(5,0))      
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[1], axes=(0,0)), R[1], axes=(5,0))      
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,2:3], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[6], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,3:], axes=([2,3,6],[3,1,0]))
	
		p = np.tensordot(np.tensordot(L[2], GV[3], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))
		
		p = np.tensordot(np.tensordot(L[2], GV[4], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,2:3], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[2], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,1:2], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[2], GV[5], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,1:2], axes=([2,3,6],[3,1,0]))

		w1.extend(list(S[0,:,0,:,0,1].flatten()))

		S = 0
	
		p = np.tensordot(np.tensordot(L[1], GV[0], axes=(0,0)), R[0], axes=(5,0))      
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0])) 

		p = np.tensordot(np.tensordot(L[1], GV[1], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,2:3], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[1], GV[6], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,3:], axes=([2,3,6],[3,1,0]))
	
		p = np.tensordot(np.tensordot(L[3], GV[3], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[3], GV[4], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,2:3], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[1], GV[2], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,1:2], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[3], GV[5], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,1:2], axes=([2,3,6],[3,1,0]))

		w1.extend(list(S[0,:,0,:,1,0].flatten()))
		w1.extend(list(S[0,:,0,:,1,2].flatten()))

		S = 0
	
		p = np.tensordot(np.tensordot(L[1], GV[0], axes=(0,0)), R[1], axes=(5,0))      
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0])) 

		p = np.tensordot(np.tensordot(L[1], GV[1], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,2:3], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[1], GV[6], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,3:], axes=([2,3,6],[3,1,0]))
	
		p = np.tensordot(np.tensordot(L[3], GV[3], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[3], GV[4], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,2:3], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[1], GV[2], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,1:2], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[3], GV[5], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,1:2], axes=([2,3,6],[3,1,0]))

		w1.extend(list(S[0,:,0,:,1,1].flatten()))
		w1.extend(list(S1[0,:,0,:,1,3].flatten()))
		
		return w1 

	def Hv_As(L, R, GV, wi, wj):
		#[B0r1r0,B0r1e1,B0r3r0,B0r3e1,B1r4r0,B1r4e1,B0r2e0]
		S = 0
		w1=[]
		
		p = np.tensordot(np.tensordot(L[0], GV[0], axes=(0,0)), R[0], axes=(5,0))      #[vR] wL wR vR*  [vL] d d vR = wL wR vR* d d [vR]  [vL] wL wR vL* = wL wR vR* d d wL wR vL*
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))   # wL [wR] vR* [d] d wL wR vL*  [wL] wR d [d*] = wL vR* [d] [wL] wR vL* [wR] d  [wL] [wR] d [d*] = wL vR* wR vL* d d

		p = np.tensordot(np.tensordot(L[0], GV[2], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,2:3], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[6], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:2], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[2], GV[5], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,3:], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[2], GV[4], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,3:], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))
	
		p = np.tensordot(np.tensordot(L[0], GV[1], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[3], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,2:3], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))
	
		S1 = S.copy()
		w1.extend(list(S[0,:,0,:,0,0].flatten()))

		S = 0
	
		p = np.tensordot(np.tensordot(L[0], GV[0], axes=(0,0)), R[1], axes=(5,0))      
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[2], axes=(0,0)), R[1], axes=(5,0))      
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,2:3], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[6], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:2], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))
	
		p = np.tensordot(np.tensordot(L[2], GV[4], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,3:], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))
		
		p = np.tensordot(np.tensordot(L[0], GV[1], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[3], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,2:3], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[2], GV[5], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,3:], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		w1.extend(list(S[0,:,0,:,0,1].flatten()))
		w1.extend(list(S1[0,:,0,:,2,0].flatten()))
		w1.extend(list(S[0,:,0,:,2,1].flatten()))

		S = 0
	
		p = np.tensordot(np.tensordot(L[1], GV[0], axes=(0,0)), R[0], axes=(5,0))      
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0])) 

		p = np.tensordot(np.tensordot(L[1], GV[2], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,2:3], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[1], GV[6], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:2], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))
	
		p = np.tensordot(np.tensordot(L[3], GV[4], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,3:], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[1], GV[1], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[1], GV[3], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,2:3], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[3], GV[5], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,3:], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		w1.extend(list(S[0,:,0,:,3,0].flatten()))

		S = 0
	
		p = np.tensordot(np.tensordot(L[1], GV[0], axes=(0,0)), R[1], axes=(5,0))      
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0])) 

		p = np.tensordot(np.tensordot(L[1], GV[2], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,2:3], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[1], GV[6], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:2], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))
	
		p = np.tensordot(np.tensordot(L[3], GV[4], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,3:], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[1], GV[1], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[1], GV[3], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,2:3], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[3], GV[5], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,3:], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		w1.extend(list(S[0,:,0,:,3,1].flatten()))
		w1.extend(list(S1[0,:,0,:,1,1].flatten()))
		
		return w1 

	def Hv_sB(L, R, GV, wi, wj):
		#[B0rh20,B0rh30,B0rh40,B0rh51,B1eh20,B1eh30,B1eh40,B1eh51,B0eh10]
		S = 0
		w1=[]
		
		p = np.tensordot(np.tensordot(L[0], GV[0], axes=(0,0)), R[0], axes=(5,0))      #[vR] wL wR vR*  [vL] d d vR = wL wR vR* d d [vR]  [vL] wL wR vL* = wL wR vR* d d wL wR vL*
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,1:2], axes=([2,3,6],[3,1,0]))   # wL [wR] vR* [d] d wL wR vL*  [wL] wR d [d*] = wL vR* [d] [wL] wR vL* [wR] d  [wL] [wR] d [d*] = wL vR* wR vL* d d

		p = np.tensordot(np.tensordot(L[0], GV[1], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,2:3], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[2], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,3:4], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[8], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[2], GV[7], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,4:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[2], GV[4], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,1:2], axes=([2,3,6],[3,1,0]))
	
		p = np.tensordot(np.tensordot(L[2], GV[5], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,2:3], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[2], GV[6], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,3:4], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[3], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,4:], axes=([2,3,6],[3,1,0]))
	
		S1 = S.copy()
		w1.extend(list(S[0,:,0,:,0,1].flatten()))
		w1.extend(list(S[0,:,0,:,0,2].flatten()))
		w1.extend(list(S[0,:,0,:,0,3].flatten()))

		S = 0
	
		p = np.tensordot(np.tensordot(L[0], GV[0], axes=(0,0)), R[1], axes=(5,0))      
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,1:2], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[1], axes=(0,0)), R[1], axes=(5,0))      
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,2:3], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[2], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,3:4], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[8], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))
	
		p = np.tensordot(np.tensordot(L[2], GV[4], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,1:2], axes=([2,3,6],[3,1,0]))
		
		p = np.tensordot(np.tensordot(L[2], GV[5], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,2:3], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[2], GV[6], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,3:4], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[3], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,4:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[2], GV[7], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,4:], axes=([2,3,6],[3,1,0]))

		w1.extend(list(S[0,:,0,:,0,4].flatten()))

		S = 0
	
		p = np.tensordot(np.tensordot(L[1], GV[0], axes=(0,0)), R[0], axes=(5,0))      
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,1:2], axes=([2,3,6],[3,1,0])) 

		p = np.tensordot(np.tensordot(L[1], GV[1], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,2:3], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[1], GV[2], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,3:4], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[1], GV[8], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))
	
		p = np.tensordot(np.tensordot(L[3], GV[4], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,1:2], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[3], GV[5], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,2:3], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[3], GV[6], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,3:4], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[1], GV[3], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,4:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[3], GV[7], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,4:], axes=([2,3,6],[3,1,0]))

		w1.extend(list(S[0,:,0,:,1,1].flatten()))
		w1.extend(list(S[0,:,0,:,1,2].flatten()))
		w1.extend(list(S[0,:,0,:,1,3].flatten()))

		S = 0
	
		p = np.tensordot(np.tensordot(L[1], GV[0], axes=(0,0)), R[1], axes=(5,0))      
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,1:2], axes=([2,3,6],[3,1,0])) 

		p = np.tensordot(np.tensordot(L[1], GV[1], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,2:3], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[1], GV[2], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,3:4], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[1], GV[8], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))
	
		p = np.tensordot(np.tensordot(L[3], GV[4], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,1:2], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[3], GV[5], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,2:3], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[3], GV[6], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,3:4], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[1], GV[3], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,4:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[3], GV[7], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:], axes=([1,3],[0,3])), wj[:,:,:,4:], axes=([2,3,6],[3,1,0]))

		w1.extend(list(S[0,:,0,:,1,4].flatten()))
		w1.extend(list(S1[0,:,0,:,1,0].flatten()))
		
		return w1 

	def Hv_Bs(L, R, GV, wi, wj):
		#[B0h2r0,B0h2e1,B0h3r0,B0h3e1,B0h4r0,B0h4e1,B1h1r0,B1h1e1,B0h5e0]
		S = 0
		w1=[]
		
		p = np.tensordot(np.tensordot(L[0], GV[0], axes=(0,0)), R[0], axes=(5,0))      #[vR] wL wR vR*  [vL] d d vR = wL wR vR* d d [vR]  [vL] wL wR vL* = wL wR vR* d d wL wR vL*
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:2], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))   # wL [wR] vR* [d] d wL wR vL*  [wL] wR d [d*] = wL vR* [d] [wL] wR vL* [wR] d  [wL] [wR] d [d*] = wL vR* wR vL* d d

		p = np.tensordot(np.tensordot(L[0], GV[2], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,2:3], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[4], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,3:4], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[8], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,4:], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[2], GV[7], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[2], GV[6], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))
	
		p = np.tensordot(np.tensordot(L[0], GV[1], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:2], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[3], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,2:3], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[5], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,3:4], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))
	
		S1 = S.copy()
		w1.extend(list(S[0,:,0,:,1,0].flatten()))

		S = 0
	
		p = np.tensordot(np.tensordot(L[0], GV[0], axes=(0,0)), R[1], axes=(5,0))      
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:2], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[2], axes=(0,0)), R[1], axes=(5,0))      
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,2:3], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[4], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,3:4], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[8], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,4:], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))
	
		p = np.tensordot(np.tensordot(L[2], GV[6], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))
		
		p = np.tensordot(np.tensordot(L[0], GV[1], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:2], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[3], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,2:3], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[5], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,3:4], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[2], GV[7], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		w1.extend(list(S[0,:,0,:,1,1].flatten()))
		w1.extend(list(S1[0,:,0,:,2,0].flatten()))
		w1.extend(list(S[0,:,0,:,2,1].flatten()))
		w1.extend(list(S1[0,:,0,:,3,0].flatten()))
		w1.extend(list(S[0,:,0,:,3,1].flatten()))

		S = 0
	
		p = np.tensordot(np.tensordot(L[1], GV[0], axes=(0,0)), R[0], axes=(5,0))      
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:2], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0])) 

		p = np.tensordot(np.tensordot(L[1], GV[2], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,2:3], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[1], GV[4], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,3:4], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[1], GV[8], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,4:], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))
	
		p = np.tensordot(np.tensordot(L[3], GV[6], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[1], GV[1], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:2], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[1], GV[3], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,2:3], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[1], GV[5], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,3:4], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[3], GV[7], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		w1.extend(list(S[0,:,0,:,0,0].flatten()))

		S = 0
	
		p = np.tensordot(np.tensordot(L[1], GV[0], axes=(0,0)), R[1], axes=(5,0))      
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:2], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0])) 

		p = np.tensordot(np.tensordot(L[1], GV[2], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,2:3], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[1], GV[4], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,3:4], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[1], GV[8], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,4:], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))
	
		p = np.tensordot(np.tensordot(L[3], GV[6], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[1], GV[1], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:2], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[1], GV[3], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,2:3], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[1], GV[5], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,3:4], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[3], GV[7], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,1:], axes=([2,3,6],[3,1,0]))

		w1.extend(list(S[0,:,0,:,0,1].flatten()))
		w1.extend(list(S1[0,:,0,:,4,1].flatten()))
		
		return w1

	'''
	Convert the ground state eigenvector obtained from the Lanczos algorithm into Matrix Product State (MPS) block-form tensors. This transformation allows the wavefunction to be efficiently represented and manipulated within the DMRG framework.
	'''
	def FS_ss(gv,EV,E,td):
		l0 = gv[0].shape[0]
		l1 = gv[2].shape[0]
		r0 = gv[0].shape[3]
		r1 = gv[1].shape[3]
	
		B0 = np.zeros((l0+l1,r0+r1))      
		B1 = np.zeros((l0, r0))       

		B0[:l0,:r0] = EV[:l0*r0].reshape((l0,r0))
		B0[:l0,r0:] = EV[l0*r0 : l0*r0+l0*r1].reshape((l0,r1))
		B0[l0:,:r0] = EV[l0*r0+l0*r1 : l0*r0+l0*r1+l1*r0].reshape((l1,r0))
		B0[l0:,r0:] = EV[l0*r0+l0*r1+l1*r0 : l0*r0+l0*r1+l1*r0+l1*r1].reshape((l1,r1))

		B1 = EV[l0*r0+l0*r1+l1*r0+l1*r1 :].reshape((l0,r0))

		U0, S0, V0 = np.linalg.svd(B0, full_matrices=False)
		U1, S1, V1 = np.linalg.svd(B1, full_matrices=False)       # SVD calculates V dagger ; we need V dagger
	
		marker = {}
		for i in S0:
			marker[i] = 0
		for i in S1:
			marker[i] = 1

		chi = np.sum(np.hstack((S0,S1))>1.e-8) 
		#print(colored(chi,'red'))
		if(chi > td):
			chi_ = td
		else:
			chi_ = chi
	
		S = np.sort(np.hstack((S0,S1)))[::-1][:chi_]

		s0=[]
		s1=[]
		smod = 0
		for i in S:
			smod=smod+i**2
			if(marker[i]==0):
				s0.append(i)
			else:
				s1.append(i)
		if(s1==[]):
			s1.append(max(S1))
			#print (colored('ProblemS1', 'red'))
		if(s0==[]):
			s0.append(max(S0))
			#print (colored('ProblemS0', 'red'))
		#print(colored('%f'%smod, 'red'))

		mi00 = np.zeros((l0,1,len(s0)))
		mi01 = np.zeros((l0,1,len(s1)))
		mi10 = np.zeros((l1,1,len(s0)))

		mj00 = np.zeros((len(s0),1,r0))
		mj01 = np.zeros((len(s0),1,r1))
		mj10 = np.zeros((len(s1),1,r0))

		mi00[:,0,:] = U0[:l0,:len(s0)]
		mi01[:,0,:] = U1[:,:len(s1)]
		mi10[:,0,:] = U0[l0:,:len(s0)]

		mj00[:,0,:] = V0[:len(s0),:r0]
		mj01[:,0,:] = V0[:len(s0),r0:]
		mj10[:,0,:] = V1[:len(s1),:]

		#print(len(s0),len(s1))
		s0.extend(s1)
		S = np.diag(s0)   #S

		return (E,[mi00, mi01, mi10],[mj00, mj01, mj10],S)

	def FS_sA(gv,EV,E,td):
		l0 = gv[0].shape[0]
		l1 = gv[3].shape[0]
		r0 = gv[0].shape[3]
		r1 = gv[2].shape[3]
	
		B0 = np.zeros((l0+l1,2*r0+r1))      
		B1 = np.zeros((l0, r0))       

		B0[:l0,:r0] = EV[:l0*r0].reshape((l0,r0))
		B0[:l0,r0:2*r0] = EV[l0*r0 : l0*r0+l0*r0].reshape((l0,r0))
		B0[:l0,2*r0:] = EV[2*l0*r0 : 2*l0*r0+l0*r1].reshape((l0,r1))
		B0[l0:,:r0] = EV[2*l0*r0+l0*r1 : 2*l0*r0+l0*r1+l1*r0].reshape((l1,r0))
		B0[l0:,r0:2*r0] = EV[2*l0*r0+l0*r1+l1*r0 : 2*l0*r0+l0*r1+l1*r0+l1*r0].reshape((l1,r0))
		B0[l0:,2*r0:] = EV[2*l0*r0+l0*r1+l1*r0+l1*r0 : 2*l0*r0+l0*r1+l1*r0+l1*r0+l1*r1].reshape((l1,r1))

		B1 = EV[2*l0*r0+l0*r1+l1*r0+l1*r0+l1*r1 :].reshape((l0,r0))

		U0, S0, V0 = np.linalg.svd(B0, full_matrices=False)
		U1, S1, V1 = np.linalg.svd(B1, full_matrices=False)       # SVD calculates V dagger ; we need V dagger
		#print(S0,S1)

		marker = {}
		for i in S0:
			marker[i] = 0
		for i in S1:
			marker[i] = 1

		chi = np.sum(np.hstack((S0,S1))>1.e-8) 
		#print(colored(chi,'red'))
		if(chi > td):
			chi_ = td
		else:
			chi_ = chi
	
		S = np.sort(np.hstack((S0,S1)))[::-1][:chi_]
		s0=[]
		s1=[]
		smod = 0
		for i in S:
			smod=smod+i**2
			if(marker[i]==0):
				s0.append(i)
			else:
				s1.append(i)

		if(s1==[]):
			s1.append(max(S1))
			#print (colored('ProblemS1', 'red'))
		if(s0==[]):
			s0.append(max(S0))
			#print (colored('ProblemS0', 'red'))
		#print(colored('%f'%smod, 'red'))

		mi00 = np.zeros((l0,1,len(s0)))
		mi01 = np.zeros((l0,1,len(s1)))
		mi10 = np.zeros((l1,1,len(s0)))

		mj0r10 = np.zeros((len(s0),1,r0))
		mj0r30 = np.zeros((len(s0),1,r0))
		mj0r21 = np.zeros((len(s0),1,r1))
		mj1r40 = np.zeros((len(s1),1,r0))

		mi00[:,0,:] = U0[:l0,:len(s0)]
		mi01[:,0,:] = U1[:,:len(s1)]
		mi10[:,0,:] = U0[l0:,:len(s0)]

		mj0r10[:,0,:] = V0[:len(s0),:r0]
		mj0r30[:,0,:] = V0[:len(s0),r0:2*r0]
		mj0r21[:,0,:] = V0[:len(s0),2*r0:]
		mj1r40[:,0,:] = V1[:len(s1),:]

		#print(len(s0),len(s1))
		s0.extend(s1)
		S = np.diag(s0)   #S

		return (E,[mi00, mi01, mi10],[mj0r10, mj0r30, mj0r21, mj1r40],S)

	def FS_sB(gv,EV,E,td):
		l0 = gv[0].shape[0]
		l1 = gv[4].shape[0]
		r0 = gv[0].shape[3]
		r1 = gv[3].shape[3]
	
		B0 = np.zeros((l0+l1,3*r0+r1))      
		B1 = np.zeros((l0, r0))       

		B0[:l0,:r0] = EV[:l0*r0].reshape((l0,r0))
		B0[:l0,r0:2*r0] = EV[l0*r0 : 2*l0*r0].reshape((l0,r0))
		B0[:l0,2*r0:3*r0] = EV[2*l0*r0 : 3*l0*r0].reshape((l0,r0))
		B0[:l0,3*r0:] = EV[3*l0*r0 : 3*l0*r0+l0*r1].reshape((l0,r1))
		B0[l0:,:r0] = EV[3*l0*r0+l0*r1 : 3*l0*r0+l0*r1+l1*r0].reshape((l1,r0))
		B0[l0:,r0:2*r0] = EV[3*l0*r0+l0*r1+l1*r0 : 3*l0*r0+l0*r1+l1*r0+l1*r0].reshape((l1,r0))
		B0[l0:,2*r0:3*r0] = EV[3*l0*r0+l0*r1+l1*r0+l1*r0 : 3*l0*r0+l0*r1+l1*r0+l1*r0+l1*r0].reshape((l1,r0))
		B0[l0:,3*r0:] = EV[3*l0*r0+l0*r1+l1*r0+l1*r0+l1*r0 : 3*l0*r0+l0*r1+l1*r0+l1*r0+l1*r0+l1*r1].reshape((l1,r1))

		B1 = EV[3*l0*r0+l0*r1+l1*r0+l1*r0+l1*r0+l1*r1 :].reshape((l0,r0))

		U0, S0, V0 = np.linalg.svd(B0, full_matrices=False)
		U1, S1, V1 = np.linalg.svd(B1, full_matrices=False)       # SVD calculates V dagger ; we need V dagger
	
		marker = {}
		for i in S0:
			marker[i] = 0
		for i in S1:
			marker[i] = 1

		chi = np.sum(np.hstack((S0,S1))>1.e-8) 
		#print(colored(chi,'red'))
		if(chi > td):
			chi_ = td
		else:
			chi_ = chi
	
		S = np.sort(np.hstack((S0,S1)))[::-1][:chi_]
		s0=[]
		s1=[]
		smod = 0
		for i in S:
			smod=smod+i**2
			if(marker[i]==0):
				s0.append(i)
			else:
				s1.append(i)
		if(s1==[]):
			s1.append(max(S1))
			#print (colored('ProblemS1', 'red'))
		if(s0==[]):
			s0.append(max(S0))
			#print (colored('ProblemS0', 'red'))
		#print(colored('%f'%smod, 'red'))

		mi00 = np.zeros((l0,1,len(s0)))
		mi01 = np.zeros((l0,1,len(s1)))
		mi10 = np.zeros((l1,1,len(s0)))

		mj0h20 = np.zeros((len(s0),1,r0))
		mj0h30 = np.zeros((len(s0),1,r0))
		mj0h40 = np.zeros((len(s0),1,r0))
		mj0h51 = np.zeros((len(s0),1,r1))
		mj1h10 = np.zeros((len(s1),1,r0))

		mi00[:,0,:] = U0[:l0,:len(s0)]
		mi01[:,0,:] = U1[:,:len(s1)]
		mi10[:,0,:] = U0[l0:,:len(s0)]

		mj0h20[:,0,:] = V0[:len(s0),:r0]
		mj0h30[:,0,:] = V0[:len(s0),r0:2*r0]
		mj0h40[:,0,:] = V0[:len(s0),2*r0:3*r0]
		mj0h51[:,0,:] = V0[:len(s0),3*r0:]
		mj1h10[:,0,:] = V1[:len(s1),:]

		#print(len(s0),len(s1))
		s0.extend(s1)
		S = np.diag(s0)   #S

		return (E,[mi00, mi01, mi10],[mj0h20, mj0h30, mj0h40, mj0h51, mj1h10],S)

	def FS_As(gv,EV,E,td):
		l0 = gv[0].shape[0]
		l1 = gv[4].shape[0]
		r0 = gv[0].shape[3]
		r1 = gv[1].shape[3]
	
		B0 = np.zeros((2*l0+l1,r0+r1))      
		B1 = np.zeros((l0, r0))       

		B0[:l0,:r0] = EV[:l0*r0].reshape((l0,r0))
		B0[:l0,r0:] = EV[l0*r0 : l0*r0+l0*r1].reshape((l0,r1))
		B0[l0:2*l0,:r0] = EV[l0*r0+l0*r1 : l0*r0+l0*r1+l0*r0].reshape((l0,r0))
		B0[l0:2*l0,r0:] = EV[l0*r0+l0*r1+l0*r0 : l0*r0+l0*r1+l0*r0+l0*r1].reshape((l0,r1))
		B0[2*l0:,:r0] = EV[l0*r0+l0*r1+l0*r0+l0*r1 : l0*r0+l0*r1+l0*r0+l0*r1+l1*r0].reshape((l1,r0))
		B0[2*l0:,r0:] = EV[l0*r0+l0*r1+l0*r0+l0*r1+l1*r0 : l0*r0+l0*r1+l0*r0+l0*r1+l1*r0+l1*r1].reshape((l1,r1))

		B1 = EV[l0*r0+l0*r1+l0*r0+l0*r1+l1*r0+l1*r1 :].reshape((l0,r0))

		#print(B0.shape)
		U0, S0, V0 = np.linalg.svd(B0, full_matrices=False)
		U1, S1, V1 = np.linalg.svd(B1, full_matrices=False)       # SVD calculates V dagger ; we need V dagger
		#print(B1.shape,l0,l1,r0,r1)

		marker = {}
		for i in S0:
			marker[i] = 0
		for i in S1:
			marker[i] = 1

		chi = np.sum(np.hstack((S0,S1))>1.e-8) 
		#print(colored(chi,'red'))
		if(chi > td):
			chi_ = td
		else:
			chi_ = chi
	
		S = np.sort(np.hstack((S0,S1)))[::-1][:chi_]
		s0=[]
		s1=[]
		smod = 0
		for i in S:
			smod=smod+i**2
			if(marker[i]==0):
				s0.append(i)
			else:
				s1.append(i)
		#print(S0,S1)
		if(s1==[]):
			s1.append(max(S1))
			#print (colored('ProblemS1', 'red'))
		if(s0==[]):
			s0.append(max(S0))
			#print (colored('ProblemS0', 'red'))
		#print(colored('%f'%smod, 'red'))

		mi0r10 = np.zeros((l0,1,len(s0)))
		mi0r30 = np.zeros((l0,1,len(s0)))
		mi0r21 = np.zeros((l0,1,len(s1)))
		mi1r40 = np.zeros((l1,1,len(s0)))

		mj00 = np.zeros((len(s0),1,r0))
		mj01 = np.zeros((len(s0),1,r1))
		mj10 = np.zeros((len(s1),1,r0))
			
		mi0r10[:,0,:] = U0[:l0,:len(s0)]
		mi0r30[:,0,:] = U0[l0:2*l0,:len(s0)]
		mi0r21[:,0,:] = U1[:,:len(s1)]
		mi1r40[:,0,:] = U0[2*l0:,:len(s0)]

		mj00[:,0,:] = V0[:len(s0),:r0]
		mj01[:,0,:] = V0[:len(s0),r0:]
		mj10[:,0,:] = V1[:len(s1),:]

		#print(len(s0),len(s1))
		s0.extend(s1)
		S = np.diag(s0)   #S

		return (E,[mi0r10, mi0r30, mi0r21, mi1r40],[mj00, mj01, mj10],S)
	
	def FS_Bs(gv,EV,E,td):
		l0 = gv[0].shape[0]
		l1 = gv[6].shape[0]
		r0 = gv[0].shape[3]
		r1 = gv[1].shape[3]
	
		B0 = np.zeros((3*l0+l1,r0+r1))      
		B1 = np.zeros((l0, r0))       

		B0[:l0,:r0] = EV[:l0*r0].reshape((l0,r0))
		B0[:l0,r0:] = EV[l0*r0 : l0*r0+l0*r1].reshape((l0,r1))
		B0[l0:2*l0,:r0] = EV[l0*r0+l0*r1 : l0*r0+l0*r1+l0*r0].reshape((l0,r0))
		B0[l0:2*l0,r0:] = EV[l0*r0+l0*r1+l0*r0 : l0*r0+l0*r1+l0*r0+l0*r1].reshape((l0,r1))
		B0[2*l0:3*l0,:r0] = EV[l0*r0+l0*r1+l0*r0+l0*r1 : l0*r0+l0*r1+l0*r0+l0*r1+l0*r0].reshape((l0,r0))
		B0[2*l0:3*l0,r0:] = EV[l0*r0+l0*r1+l0*r0+l0*r1+l0*r0 : l0*r0+l0*r1+l0*r0+l0*r1+l0*r0+l0*r1].reshape((l0,r1))
		B0[3*l0:,:r0] = EV[l0*r0+l0*r1+l0*r0+l0*r1+l0*r0+l0*r1 : l0*r0+l0*r1+l0*r0+l0*r1+l0*r0+l0*r1+l1*r0].reshape((l1,r0))
		B0[3*l0:,r0:] = EV[l0*r0+l0*r1+l0*r0+l0*r1+l0*r0+l0*r1+l1*r0 : l0*r0+l0*r1+l0*r0+l0*r1+l0*r0+l0*r1+l1*r0+l1*r1].reshape((l1,r1))

		B1 = EV[l0*r0+l0*r1+l0*r0+l0*r1+l0*r0+l0*r1+l1*r0+l1*r1 :].reshape((l0,r0))

		U0, S0, V0 = np.linalg.svd(B0, full_matrices=False)
		U1, S1, V1 = np.linalg.svd(B1, full_matrices=False)       # SVD calculates V dagger ; we need V dagger
	
		marker = {}
		for i in S0:
			marker[i] = 0
		for i in S1:
			marker[i] = 1

		chi = np.sum(np.hstack((S0,S1))>1.e-8)
		#print(colored(chi,'red')) 
		if(chi > td):
			chi_ = td
		else:
			chi_ = chi
	
		S = np.sort(np.hstack((S0,S1)))[::-1][:chi_]
		s0=[]
		s1=[]
		smod = 0
		for i in S:
			smod=smod+i**2
			if(marker[i]==0):
				s0.append(i)
			else:
				s1.append(i)
		if(s1==[]):
			s1.append(max(S1))
			#print (colored('ProblemS1', 'red'))
		if(s0==[]):
			s0.append(max(S0))
			#print (colored('ProblemS0', 'red'))
		#print(colored('%f'%smod, 'red'))

		mi0h20 = np.zeros((l0,1,len(s0)))
		mi0h30 = np.zeros((l0,1,len(s0)))
		mi0h40 = np.zeros((l0,1,len(s0)))
		mi0h51 = np.zeros((l0,1,len(s1)))
		mi1h10 = np.zeros((l1,1,len(s0)))

		mj00 = np.zeros((len(s0),1,r0))
		mj01 = np.zeros((len(s0),1,r1))
		mj10 = np.zeros((len(s1),1,r0))

		mi0h20[:,0,:] = U0[:l0,:len(s0)]
		mi0h30[:,0,:] = U0[l0:2*l0,:len(s0)]
		mi0h40[:,0,:] = U0[2*l0:3*l0,:len(s0)]
		mi0h51[:,0,:] = U1[:,:len(s1)]
		mi1h10[:,0,:] = U0[3*l0:,:len(s0)]

		mj00[:,0,:] = V0[:len(s0),:r0]
		mj01[:,0,:] = V0[:len(s0),r0:]
		mj10[:,0,:] = V1[:len(s1),:]

		#print(len(s0),len(s1))
		s0.extend(s1)
		S = np.diag(s0)   #S

		return (E,[mi0h20, mi0h30, mi0h40, mi0h51, mi1h10],[mj00, mj01, mj10],S)






 
