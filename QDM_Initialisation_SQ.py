
import matplotlib.pylab as plt
from termcolor import colored
from collections import deque
from DimerDensity import dimerDen as DD
import itertools as it
import numpy as np
import pickle
import time

N = 20
d=2

########################### Initialisation ############################

M = {}
d0 = 1
d1 = 1

# Right Normalised - Exact

B0 = np.zeros([d0,1])
B1 = np.zeros([d1,1])
B0[0,0] = 1 
B1[0,0] = 1
M[N-1] = [B0, B1] 

for i in range(N-2,N-6,-1):
	I = np.eye(d0+d1)
	B00 = np.zeros([d0+d1, 1, d0])
	B01 = np.zeros([d0+d1, 1, d1])
	B10 = np.zeros([d0, 1, d0])
	B00[:,0,:] = I[:,:d0]
	B01[:,0,:] = I[:,d0:]
	B10[:,0,:] = np.eye(d0)
	M[i] = [B00, B01, B10]
	dt = d0+d1
	d1 = d0
	d0 = dt

'''
B0 = np.array([[1]])
B1 = np.array([[1]])
M[N-1] = [B0, B1] 

B00 = np.zeros([2, 1, 1])
B01 = np.zeros([2, 1, 1])
B10 = np.zeros([1, 1, 1])
B00[:,0,:] = np.array([[1],[0]])
B01[:,0,:] = np.array([[0],[1]])
B10[:,0,:] = np.array([[1]])
M[N-2] = [B00, B01, B10]

B00 = np.zeros([3, 1, 2])
B01 = np.zeros([3, 1, 1])
B10 = np.zeros([2, 1, 2])

B00[:,0,:] = np.array([[1,0],[0,1],[0,0]])
B01[:,0,:] = np.array([[0],[0],[1]])
B10[:,0,:] = np.array([[1,0],[0,1]])
M[N-3] = [B00, B01, B10]

B00 = np.zeros([5, 1, 3])
B01 = np.zeros([5, 1, 2])
B10 = np.zeros([3, 1, 3])

B00[:,0,:] = np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,0],[0,0,0]])
B01[:,0,:] = np.array([[0,0],[0,0],[0,0],[1,0],[0,1]])
B10[:,0,:] = np.array([[1,0,0],[0,1,0],[0,0,1]])
M[N-4] = [B00, B01, B10]

B00 = np.zeros([8, 1, 5])
B01 = np.zeros([8, 1, 3])
B10 = np.zeros([5, 1, 5])

B00[:,0,:] = np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
B01[:,0,:] = np.array([[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
B10[:,0,:] = np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])
M[N-5] = [B00, B01, B10]
'''

# Right Normalised - Random

for i in range(6,N-5):
	
	a1 = np.random.uniform(-1,1,(d0,d0+d1))       #8,13))
	a2 = np.random.uniform(-1,1,(d1,d0))        #5,8))
	
	_,_,V1 = np.linalg.svd(a1, full_matrices=False)
	_,_,M3 = np.linalg.svd(a2, full_matrices=False)

	B00 = np.zeros([d0,1,d0])     #8, 1, 8])
	B01 = np.zeros([d0,1,d1])      #8, 1, 5])
	B10 = np.zeros([d1,1,d0])      #5, 1, 8])
	
	B00[:,0,:]=V1[:,:d0]       #:,:8]
	B01[:,0,:]=V1[:,d0:]       #:,8:]
	B10[:,0,:]=M3
	
	M[i] = [B00, B01, B10] 

# Left Normalised - Exact

d0 = 1
d1 = 1

B0 = np.zeros([1,d0])
B1 = np.zeros([1,d1])
B0[0,0] = 1 
B1[0,0] = 1
M[0] = [B0, B1] 

for i in range(1,5):
	I = np.eye(d0+d1)
	B00 = np.zeros([d0, 1, d0+d1])
	B01 = np.zeros([d0, 1, d0])
	B10 = np.zeros([d1, 1, d0+d1])
	B00[:,0,:] = I[:d0,:]
	B01[:,0,:] = np.eye(d0)
	B10[:,0,:] = I[d0:,:]
	M[i] = [B00, B01, B10]
	dt = d0+d1
	d1 = d0
	d0 = dt
	
'''
B0 = np.array([[1]])
B1 = np.array([[1]])
M[0] = [B0, B1] 

B00 = np.zeros([1, 1, 2])
B01 = np.zeros([1, 1, 1])
B10 = np.zeros([1, 1, 2])
B00[:,0,:] = np.array([[1,0]])
B01[:,0,:] = np.array([[1]])
B10[:,0,:] = np.array([[0,1]])
M[1] = [B00, B01, B10]

B00 = np.zeros([2, 1, 3])
B01 = np.zeros([2, 1, 2])
B10 = np.zeros([1, 1, 3])
B00[:,0,:] = np.array([[1,0,0],[0,1,0]])
B01[:,0,:] = np.array([[1,0],[0,1]])
B10[:,0,:] = np.array([[0,0,1]])
M[2] = [B00, B01, B10]

B00 = np.zeros([3, 1, 5])
B01 = np.zeros([3, 1, 3])
B10 = np.zeros([2, 1, 5])
B00[:,0,:] = np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0]])
B01[:,0,:] = np.array([[1,0,0],[0,1,0],[0,0,1]])
B10[:,0,:] = np.array([[0,0,0,1,0],[0,0,0,0,1]])
M[3] = [B00, B01, B10]

B00 = np.zeros([5, 1, 8])
B01 = np.zeros([5, 1, 5])
B10 = np.zeros([3, 1, 8])
B00[:,0,:] = np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0]])
B01[:,0,:] = np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])
B10[:,0,:] = np.array([[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]])
M[4] = [B00, B01, B10]
'''

# Left Normalised - Random

a1 = np.random.uniform(-1,1,(d0+d1,d0))    #13,8))
a2 = np.random.uniform(-1,1,(d0,d1))     #8,5))
	
V1,_,_ = np.linalg.svd(a1, full_matrices=False)
M3,_,_ = np.linalg.svd(a2, full_matrices=False)

B00 = np.zeros([d0,1,d0])     #8, 1, 8])
B01 = np.zeros([d0,1,d1])      #8, 1, 5])
B10 = np.zeros([d1,1,d0])      #5, 1, 8])

B00[:,0,:]=V1[:d0,:]        #:8,:]
B01[:,0,:]=M3
B10[:,0,:]=V1[d0:,:]        #8:,:]

M[5] = [B00, B01, B10] 

M_start = M.copy()

################################ H_MPOs ###################################

def MPO(J,Vr,Vl):
	# rung dimer - 0, empty - 1
	#J = 1
	#Vr = 5
	#Vl = 6
	Sp = np.array([[0., 1.], [0., 0.]])
	Sm = np.array([[0., 0.], [1., 0.]])
	I = np.eye(2)
	n = np.matmul(Sp,Sm)	

	W = [0]* N

	for i in range(1,N-1):
		w = np.zeros((10, 10, d, d))
		w[0,0] = I
		w[1,0] = Sm
		w[2,0] = Sp
		w[3,0] = n
		w[4,0] = I-n
		w[5,0] = n
		w[9,1] = -J*Sm
		w[9,2] = -J*Sp
		w[9,3] = Vr*n
		w[9,7] = Vl*n
		w[9,8] = Vl*(I-n)
		w[9,9] = I
		w[5,4] = I-n
		w[6,5] = I-n
		w[7,6] = I-n
		w[8,7] = I-n
	
		W[i] = w

	w = np.zeros((1, 10, d, d))
	w[0,1] = -J*Sm      
	w[0,2] = -J*Sp      
	w[0,3] = Vr*n       
	w[0,6] = Vl*(I-n)   
	w[0,7] = Vl*n       
	w[0,8] = Vl*(I-n)   
	w[0,9] = I

	W[0] = w

	w = np.zeros((10, 1, d, d))
	w[0,0] = I
	w[1,0] = Sm
	w[2,0] = Sp
	w[3,0] = n
	w[4,0] = (I-n)
	w[5,0] = n
	w[6,0] = (I-n)

	W[N-1] = w

	return W

'''
W = [0]* N
I = np.eye(2)
for i in range(1,N-1):
	w = np.zeros((1, 1, d, d))
	for j in range(1):
		w[j,j]=I
	W[i] = w

w = np.zeros((1, 1, d, d))
w[0,0]=I
W[0]=w
w = np.zeros((1, 1, d, d))
w[0,0]=I
W[N-1]=w
'''

################################ Initial LE and RE ###################################

# up = d
# down = d*
# MPO - wL,wR,d,d*

### LE ###

def LEnv(i,M,W,LE):
	le=[]
	b0 = [['0 0', '0 2', '2 0', '2 2'], ['0 1', '*', '2 1', '*'], ['1 0', '1 2', '*', '*'], ['1 1', '*', '*', '*']]
	for j in range (4):
		s=0
		for k in range(4):
			if(b0[j][k]!='*'):
				b, c = b0[j][k].split()
				A = np.tensordot(M[i][int(b)], LE[i-1][k], axes=(0,0))              # [vL] d vR  [vR] wL wR vR*                 
				if(b=='0'):
					A = np.tensordot(A, W[i][:,:,:,:1], axes=([0,3],[3,0]))      # [d] vR wL [wR] vR*  [wL] wR d [d*]     
				else:		
					A = np.tensordot(A, W[i][:,:,:,1:], axes=([0,3],[3,0]))    
				
				if(c=='0'):
					A = np.tensordot(A[:,:,:,:,:1], M[i][int(c)], axes=([2,4],[0,1]))   # vR wL [vR*] wR [d]  [vL*] [d*] vR* == vR wL wR vR*    
				else:
					A = np.tensordot(A[:,:,:,:,1:], M[i][int(c)], axes=([2,4],[0,1]))
				s=s+A
		le.append(s)	
	return le


def LE_(M,W):
	for i in range(6):
		if(i == 0):
			LE = {}

			X00 = np.tensordot(M[0][0], W[0][:,:,:,:1], axes=(0,3))       #[d] vR       wL wR d [d*]
			X00 = np.tensordot(X00[:,:,:,:1], M[0][0], axes=(3,0))        #vR wL wR [d]  [d*] vR*    ==  vR wL wR vR*

			X01 = np.tensordot(M[0][0], W[0][:,:,:,:1], axes=(0,3))       
			X01 = np.tensordot(X01[:,:,:,1:], M[0][1], axes=(3,0))       

			X10 = np.tensordot(M[0][1], W[0][:,:,:,1:], axes=(0,3))      
			X10 = np.tensordot(X10[:,:,:,:1], M[0][0], axes=(3,0))       

			X11 = np.tensordot(M[0][1], W[0][:,:,:,1:], axes=(0,3))     
			X11 = np.tensordot(X11[:,:,:,1:], M[0][1], axes=(3,0))  

			LE[0] = [X00,X01,X10,X11]
		else:
			LE[i] = LEnv(i,M,W,LE)
	return LE

### RE ###

def REnv(i,M,W,RE):
	re=[]
	b0 = [['0 0', '0 1', '1 0', '1 1'], ['0 2', '*', '1 2', '*'], ['2 0', '2 1', '*', '*'], ['2 2', '*', '*', '*']]
	for j in range (4):
		s=0
		for k in range(4):
			if(b0[j][k]!='*'):
				b, c = b0[j][k].split()   
				A = np.tensordot(M[i][int(b)], RE[i+1][k], axes=(2,0))              # vL d [vR]   [vL] wL wR vL*          
				if(b=='0'):
					A = np.tensordot(A, W[i][:,:,:,:1], axes=([1,2],[3,1]))      # vL [d] [wL] wR vL*   wL [wR] d [d*]      
				else:
					A = np.tensordot(A, W[i][:,:,:,1:], axes=([1,2],[3,1]))    
				
				if(c=='0'):
					A = np.tensordot(A[:,:,:,:,:1], M[i][int(c)], axes=([2,4],[2,1]))   # vL wR [vL*] wL [d]   vL* [d*] [vR*]  == vL wR wL vL*  

				else:
					A = np.tensordot(A[:,:,:,:,1:], M[i][int(c)], axes=([2,4],[2,1]))
			
				A = np.swapaxes(A,1,2)   #vL wL wR vL*  
				s=s+A
		re.append(s)	
	return re

def RE_(M,W):
	for i in range(N-1,6,-1):    #6
		if(i == N-1):
			RE = {}

			X00 = np.tensordot(M[N-1][0], W[N-1][:,:,:,:1], axes=(1,3))       #vL [d]       wL wR d [d*]
			X00 = np.tensordot(X00[:,:,:,:1], M[N-1][0], axes=(3,1))          #vL wL wR [d]  vL* [d*]   == vL wL wR vL*

			X01 = np.tensordot(M[N-1][0], W[N-1][:,:,:,:1], axes=(1,3))       
			X01 = np.tensordot(X01[:,:,:,1:], M[N-1][1], axes=(3,1))        

			X10 = np.tensordot(M[N-1][1], W[N-1][:,:,:,1:], axes=(1,3))      
			X10 = np.tensordot(X10[:,:,:,:1], M[N-1][0], axes=(3,1))       

			X11 = np.tensordot(M[N-1][1], W[N-1][:,:,:,1:], axes=(1,3))       
			X11 = np.tensordot(X11[:,:,:,1:], M[N-1][1], axes=(3,1)) 

			RE[N-1] = [X00,X01,X10,X11]
		else:
			RE[i] = REnv(i,M,W,RE)
	return RE

##################################  Guess state  ##########################################

def guess_vector_RS(i,ss,M):
	l0 = M[i-1][0].shape[2]
	r0 = M[i][0].shape[0]
	B0rr0 = np.tensordot(np.tensordot(ss[:l0,:r0], M[i][0], axes=(1,0)), M[i+1][0], axes=(2,0))    # vL d [vR] [vL] d vR = vL d d vR
	B0re1 = np.tensordot(np.tensordot(ss[:l0,:r0], M[i][0], axes=(1,0)), M[i+1][1], axes=(2,0))
	B1er0 = np.tensordot(np.tensordot(ss[l0:,r0:], M[i][2], axes=(1,0)), M[i+1][0], axes=(2,0))
	B1ee1 = np.tensordot(np.tensordot(ss[l0:,r0:], M[i][2], axes=(1,0)), M[i+1][1], axes=(2,0))
	B0ee0 = np.tensordot(np.tensordot(ss[:l0,:r0], M[i][1], axes=(1,0)), M[i+1][2], axes=(2,0))
	
	return [B0rr0, B0re1, B1er0, B1ee1, B0ee0]

def guess_vector_LS(i,ss,M):
	l0 = M[i+1][0].shape[2]
	r0 = M[i+2][0].shape[0]
	B0rr0 = np.tensordot(np.tensordot(M[i][0], M[i+1][0], axes=(2,0)), ss[:l0,:r0], axes=(3,0))    # vL d [vR] [vL] d vR = vL d d vR
	B0re1 = np.tensordot(np.tensordot(M[i][0], M[i+1][1], axes=(2,0)), ss[l0:,r0:], axes=(3,0))
	B1er0 = np.tensordot(np.tensordot(M[i][2], M[i+1][0], axes=(2,0)), ss[:l0,:r0], axes=(3,0))
	B1ee1 = np.tensordot(np.tensordot(M[i][2], M[i+1][1], axes=(2,0)), ss[l0:,r0:], axes=(3,0))
	B0ee0 = np.tensordot(np.tensordot(M[i][1], M[i+1][2], axes=(2,0)), ss[:l0,:r0], axes=(3,0))
	
	return [B0rr0, B0re1, B1er0, B1ee1, B0ee0]

'''
def guess_vector_(i,ss,M):
	l0 = M[i][0].shape[2]
	r0 = M[i+1][0].shape[0]
	B0rr0 = np.tensordot(np.tensordot(M[i][0], ss[:l0,:r0], axes=(2,0)), M[i+1][0], axes=(2,0))    # vL d [vR] [vL] d vR = vL d d vR
	B0re1 = np.tensordot(np.tensordot(M[i][0], ss[:l0,:r0], axes=(2,0)), M[i+1][1], axes=(2,0))
	B1er0 = np.tensordot(np.tensordot(M[i][2], ss[:l0,:r0], axes=(2,0)), M[i+1][0], axes=(2,0))
	B1ee1 = np.tensordot(np.tensordot(M[i][2], ss[:l0,:r0], axes=(2,0)), M[i+1][1], axes=(2,0))
	B0ee0 = np.tensordot(np.tensordot(M[i][1], ss[l0:,r0:], axes=(2,0)), M[i+1][2], axes=(2,0))
	
	return [B0rr0, B0re1, B1er0, B1ee1, B0ee0]
'''

################################## Lanczos #################################################

def Lanczos(L, R, gv, wi, wj, td):
	
	def Hv(L, R, GV, wi, wj):
	
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

	s=0
	v=[]
	for i in range(5):
		s = s + gv[i].shape[0] * gv[i].shape[3]
		v.extend(list(gv[i][:,0,0,:].flatten()))

	V = np.zeros((s,1))
	T = np.zeros((1,1))
	v = v/np.linalg.norm(v)
	V[:,0] = v 

	w = Hv(L, R, gv, wi, wj)
	alpha = np.dot(np.conjugate(w),v)    
	w = w - alpha * v       
	
	T[0,0] = alpha  
	ev,evc = np.linalg.eigh(T)
	mev = ev[0]
	
	diff = 1
	j=1

	while(diff > 1e-13 and j<200):      
	
		beta = np.linalg.norm(w)
		if(beta == 0):
			print('danger')

		v2 = w/beta
		
		t1 = np.zeros((gv[0].shape[0],1,1,gv[0].shape[3]))
		sp = gv[0].shape[0] * gv[0].shape[3]
		t1[:,0,0,:] = v2[:sp].reshape((gv[0].shape[0], gv[0].shape[3]))    
		GV = [t1]
		for q in range(1,5):
			t = np.zeros((gv[q].shape[0],1,1,gv[q].shape[3]))
			t[:,0,0,:] = v2[sp : sp + gv[q].shape[0]*gv[q].shape[3]].reshape((gv[q].shape[0], gv[q].shape[3]))         
			sp = sp + gv[q].shape[0]*gv[q].shape[3]
			GV.append(t)

		w = Hv(L, R, GV, wi, wj)
		alpha = np.dot(np.conjugate(w),v2)     
		w = w - alpha * v2 - beta * v                       
		v = v2.copy()                  
	
		V = np.c_[V,v2]      
		T = np.c_[T,np.zeros(T.shape[0])]
		T = np.r_[T,np.zeros((1,T.shape[1]))]
		
		T[j-1,j] = beta
		T[j,j-1] = beta
		T[j,j] = alpha  
		
		eigVL,eigV = np.linalg.eigh(T)
		diff = abs(eigVL[0]-mev)	
		mev = eigVL[0]
		j=j+1

	print('LanczosDiff%i'%j, diff)
	EV = V @ eigV[:,0]                    #This need to be normalised for sum(|si|**2) = 1
	EV = EV/np.linalg.norm(EV)
	E = mev 

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
	if(chi > td):
		chi_ = td
	else:
		chi_ = chi
	
	#chi_ = len(np.hstack((S0,S1)))
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
		print (colored('Problem', 'red'))

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


##################################### Sweeps ####################################################

def sweeps(n,J,Vr,Vl):
	M = M_start.copy()
	c=0
	Eng = []
	iters = []
	W = MPO(J,Vr,Vl)
	LE = LE_(M,W)
	RE = RE_(M,W)
	SS = [ 0,0,0,0,0, np.diag([1/np.sqrt(d0+d1)]*(d0+d1)) ]    #13
	SS.extend([ 0 ]*(N-7))
	td = 100
	for i in range(n):
		for j in range(6,N-6):      
			c=c+1
			gv = guess_vector_RS(j,SS[j-1],M)
			E, M[j], M[j+1], SS[j] = Lanczos(LE[j-1], RE[j+2], gv, W[j], W[j+1],td)
	
			LE[j] = LEnv(j,M,W,LE)
			RE[j+1] = REnv(j+1,M,W,RE)
			
			Eng.append(E)
			iters.append(c)
			
			print("Sweep #%i, RS, site%i"%(i,j), E, '\n')

		td=td+100
		for j in range(N-8,4,-1):         

			c=c+1
			gv = guess_vector_LS(j,SS[j+1],M)
			E, M[j], M[j+1], SS[j] = Lanczos(LE[j-1], RE[j+2], gv, W[j], W[j+1],td)
		
			LE[j] = LEnv(j,M,W,LE)
			RE[j+1] = REnv(j+1,M,W,RE)
			
			Eng.append(E)
			iters.append(c)

			print("Sweep #%i, LS, site%i"%(i,j), E, '\n')
		td=td+100

	return Eng, iters, SS, M


##################################### Run ########################################################
#st = time.time()
#Eng, iters, SS, M = sweeps(3,J=1,Vr=6,Vl=5)
#print(Eng[-1])
#plt.plot(iters[:],Eng[:])
#plt.show()

vr = np.linspace(-5,7,15)
vl = np.linspace(-10,20,15)
points = list(it.product(vr,vl))

print(len(points))

color = {0:'#7EC0EE',1:'#EEE8AA',2:'#FF82AB',3:'#9A32CD',4:'#7CCD7C'}
c=0
for x,y in points:                 #range(len(vl)):
	print(colored(c,'blue'))

	Eng, iters, SS, M = sweeps(3,J=1,Vr=x,Vl=y)
	lad = 's'*20
	Sr,Sl,_,_,_,_,_ = DD.dimDen(lad,M,SS)

	#DD.EET(lad,SS)
	print(x,y,Sr,Sl)
	N = len(lad)
	
	if(len(Sr) == N-10 and len(Sl) == 0):
		print(colored('rungP','blue'))
		plt.plot(x, y, marker="o", markersize=5, markerfacecolor=color[0], markeredgecolor=color[0])

	elif(len(Sl) == N-10 and len(Sr) == 0):
		print(colored('colP','blue'))
		plt.plot(x, y, marker="o", markersize=5, markerfacecolor=color[1], markeredgecolor=color[1])

	elif(len(Sl) == 7 and len(Sr) == 3):
		print(colored('P3','blue'))
		plt.plot(x, y, marker="o", markersize=5, markerfacecolor=color[2], markeredgecolor=color[2])

	else:
		print(colored(len(Sr)+len(Sl), 'blue'))
		print(colored(N-10,'blue'))
		print(colored('Disorder','blue'))
		plt.plot(x, y, marker="o", markersize=5, markerfacecolor=color[4], markeredgecolor=color[4])
			
	c += 1

#plt.plot(x1,y1,c='black')

plt.plot(np.linspace(-30,30),np.linspace(-30,30),c='black')
plt.xlim(-5.5,7.5)
plt.ylim(-10.5,20.5)
plt.show()


Dgfdg










##### Entanglement Entropy on every site 'I' ################
'''
ET = []
for i in range(6, N-7):
	s =0
	print(i)
	for j in SS[i].diagonal():
		s = s + np.abs(j)**2 * (np.log(np.abs(j)**2))
	ET.append(-s)

plt.plot(np.linspace(0,18,17),ET)
plt.show()

Dsfds
'''
##### Dimer Density on every site 'I' ################

S0 = np.array([[1., 0.], [0., 0.]])
for i in range(6,N-5):
	m = M[i]
	r0 = m[0].shape[0]
	s=0
	for j in range(3):
		if(j==0):
			x = np.tensordot(np.tensordot(m[j], m[j], axes=(2,2)), S0[:1,:1], axes=([1,3],[1,0]))     #vL* d* [vR*]  vL d [vR] = vL* [d*] vL [d]  [Od] [Ou] = vL* vL
		else:
			x = np.tensordot(np.tensordot(m[j], m[j], axes=(2,2)), S0[1:,1:], axes=([1,3],[1,0])) 
		if(j!=2):
			ss = np.tensordot(SS[i-1][:r0,:r0], SS[i-1][:r0,:r0], axes=(0,0))   # [sil*] sir*  [sil] sir = sir* sir
		else:
			ss = np.tensordot(SS[i-1][r0:,r0:], SS[i-1][r0:,r0:], axes=(0,0)) 
		f = np.tensordot(ss, x, axes=([0,1],[0,1]))           # [sir*] [sir] [vL*] [vL] 
		s = s + f
	print("Site is %i "%i, s)

##### <nr(0)nr(j)> vs j : rung-rung correlation ################

Ms = M.copy()
for i in range(6,N-7):
	r0 = M[i][0].shape[2]	
	for k in range(3):
		Ms[i][0] = np.tensordot(M[i][0], SS[i][:r0,:r0], axes=(2,0))       
		Ms[i][1] = np.tensordot(M[i][1], SS[i][r0:,r0:], axes=(2,0))       
		Ms[i][2] = np.tensordot(M[i][2], SS[i][:r0,:r0], axes=(2,0))    
	
#W = np.array([[0., 0.], [0., 1.]])
W = np.eye(2)
OE = {}

X00 = np.tensordot(M[0][0], W[:1,:], axes=(0,0))       #[d] vR       [Od] Ou
X00 = np.tensordot(X00[:,:1], M[0][0], axes=(1,0))        #vR [Ou]  [d*] vR*    ==  vR vR*

X01 = np.tensordot(M[0][0], W[:1,:], axes=(0,0))       
X01 = np.tensordot(X01[:,1:], M[0][1], axes=(1,0))       

X10 = np.tensordot(M[0][1], W[1:,:], axes=(0,0))      
X10 = np.tensordot(X10[:,:1], M[0][0], axes=(1,0))       

X11 = np.tensordot(M[0][1], W[1:,:], axes=(0,0))     
X11 = np.tensordot(X11[:,1:], M[0][1], axes=(1,0)) 

def LEnv(i,m,W):
	oe=[]
	if(i != N-1):
		b0 = [['0 0', '0 2', '2 0', '2 2'], ['0 1', '*', '2 1', '*'], ['1 0', '1 2', '*', '*'], ['1 1', '*', '*', '*']]
	else:
		b0 = [['0 0'], ['0 1'], ['1 0'], ['1 1']]
	for j in range (4):
		s=0
		for k in range(len(b0[j])):
			if(b0[j][k]!='*'):
				b, c = b0[j][k].split()
				A = np.tensordot(m[i][int(b)], OE[i-1][k], axes=(0,0))              # [vL] d vR   [vR] vR*              [vL] d   [vR] vR*   == d vR*     
				if(b=='0'):
					A = np.tensordot(A, W[:1,:], axes=(0,0))      #[d] vR vR*  [Od] Ou  == vR vR* Ou                [d] vR*  [Od] Ou  == vR* Ou 
				else:		
					A = np.tensordot(A, W[1:,:], axes=(0,0))  
				
				if(i != N-1):
					if(c=='0'):
						A = np.tensordot(A[:,:,:1], m[i][int(c)], axes=([1,2],[0,1]))   # vR [vR*] [Ou]   [vL*] [d*] vR* == vR vR*          [vR*][Ou]   [vL*] [d*]
					else:
						A = np.tensordot(A[:,:,1:], m[i][int(c)], axes=([1,2],[0,1]))
				else:
					if(c=='0'):
						A = np.tensordot(A[:,:1], m[i][int(c)], axes=([0,1],[0,1]))     #[vR*][Ou]   [vL*] [d*] == 1
					else:
						A = np.tensordot(A[:,1:], m[i][int(c)], axes=([0,1],[0,1]))
				s=s+A
		oe.append(s)	
	return oe

OE[0] = [X00,X01,X10,X11]
J = []
Corr = []
for jj in range(10,200):
	for i in range(1,N):
		if(i<5 or i>N-6):
			OE[i] = LEnv(i,M,np.eye(2))  
		elif(i==6 or i==jj):
			OE[i] = LEnv(i,Ms,np.array([[1., 0.], [0., 0.]]))
		else:
			OE[i] = LEnv(i,Ms,np.eye(2))

	Corr.append(np.sum(OE[N-1]))
	J.append(jj)

#print(Corr,J)
plt.plot(J,Corr)
plt.scatter(J,Corr,s=5)
plt.yscale('log')
plt.show()
	
			
		

sdsserew


































#vR wL wR vR*
#vL wR wL vL*
#wL wR [d] d*

H = np.zeros((256,256))

c=0
for i in LE[3]:
	for j in RE[6]:
		#print(c)
		#print(i.shape,W[4].shape,W[5].shape,j.shape)
		p = np.tensordot(np.tensordot(np.tensordot(i, W[4], axes=(2,0)),W[5], axes=(3,0)),j, axes=(5,1)) 
		p = p.reshape((i.shape[3]*j.shape[3]*4 , i.shape[0]*j.shape[0]*4))
		a,b = p.shape
		#H[:a
		c=c+1   
	
hggjhjhj
#5 1 5 2 2 
#5 1 5 2 2 10 2 2 

En = []
ite = []
for i in range(len(Eng)):
	if(Eng[i] < -10):
		En.append(Eng[i])
		ite.append(iters[i])

print(len(Eng),len(iters))
plt.plot(ite[3:],En[3:])
plt.show()

'''
def lanczos(H,v):
	m = len(v)
	v = v/np.linalg.norm(v)
	V = np.zeros((len(v),m), dtype=complex)
	T = np.zeros((m,m), dtype=complex)
	V[:,0] = v

	w = np.matmul(H, v)
	alpha = np.matmul(v ,np.conjugate(np.transpose(w)))
	w = w - alpha * v
	
	T[0,0] = alpha

	for j in range(1,m):
		
		beta = np.linalg.norm(w)
		if(beta == 0):
			print('danger')

		v2 = w/beta

		w = np.matmul(H, v2)
		alpha = np.matmul(v2,np.conjugate(np.transpose(w)))
		w = w - alpha * v2 - beta * v
		v = v2.copy()
		
		V[:,j] = v2
		T[j-1,j] = beta
		T[j,j-1] = beta
		T[j,j] = alpha

	return T,V


H = np.random.rand(8,8) + np.random.rand(8,8)*1j 
H = H + H.conj().T
Hc = H.copy()

a,b = np.linalg.eigh(H) 


vg = np.random.rand(8) + np.random.rand(8)*1j 
T, V = lanczos(Hc,vg) 
A,B = np.linalg.eigh(T)


print(a)
print(A)

print(b[:,0])
print(np.matmul(V,B[:,0]))

print(a[0]*b[:,0] - np.dot(H,b[:,0]))
print(A[0]*np.dot(V,B[:,0]) - np.dot(H,np.dot(V,B[:,0])))
'''
	

	
