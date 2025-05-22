import matplotlib.pylab as plt
from termcolor import colored
import numpy as np
import pickle
import time

N = 20
d=3

########################### Initialisation ############################

M = {}
d0 = 1
d1 = 1

# Right Normalised - Exact

B0 = np.zeros([d0,2])
B1 = np.zeros([d1,1])
B0[0,0] = 1 
B0[0,1] = 1 
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

# Right Normalised - Random

for i in range(6,N-5):
	
	a1 = np.random.uniform(-1,1,(d0,d0+d1))       #8,13))
	a2 = np.random.uniform(-1,1,(d1,d0))          #5,8))
	
	_,_,V1 = np.linalg.svd(a1, full_matrices=False)
	_,_,M3 = np.linalg.svd(a2, full_matrices=False)

	B00 = np.zeros([d0,1,d0])      #8, 1, 8])
	B01 = np.zeros([d0,1,d1])      #8, 1, 5])
	B10 = np.zeros([d1,1,d0])      #5, 1, 8])
	
	B00[:,0,:]=V1[:,:d0]       #:,:8]
	B01[:,0,:]=V1[:,d0:]       #:,8:]
	B10[:,0,:]=M3
	
	M[i] = [B00, B01, B10] 

# Left Normalised - Exact

d0 = 1
d1 = 1

B0 = np.zeros([2,d0])
B1 = np.zeros([1,d1])
B0[0,0] = 1 
B0[1,0] = 1 
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


################################ H_MPOs ###################################

# rung dimer - 0, empty - 1
J = 0.01
Vr = -1
Vl = -0.01
S1 = np.array([[0., 1., 0.], [0., 0., 0.],[0., 0., 0.]])
S2 = np.array([[0., 0., 1.], [0., 0., 0.],[0., 0., 0.]])
S6 = np.array([[0., 0., 0.], [1., 0., 0.],[0., 0., 0.]])
S7 = np.array([[0., 0., 0.], [0., 0., 0.],[1., 0., 0.]])
I = np.eye(3)

def H_RS():
	W = [0]* N

	for i in range(1,N-1):
		w = np.zeros((6, 6, d, d))
		w[0,0] = I
		w[1,0] = S2
		w[2,0] = S7
		w[3,0] = I
		w[4,0] = I
		w[5,0] = 0
		w[5,1] = S1*-J
		w[5,2] = S6*-J
		w[5,3] = I*Vr
		w[5,4] = I*Vl
		w[5,5] = I
	
		W[i] = w

	w = np.zeros((1, 6, d, d))
	w[0,0] = 0
	w[0,1] = S1*-J
	w[0,2] = S6*-J
	w[0,3] = I*Vr
	w[0,4] = I*Vl
	w[0,5] = I
	W[0] = w

	w = np.zeros((6, 1, d, d))
	w[0,0] = I
	w[1,0] = S2
	w[2,0] = S7
	w[3,0] = I
	w[4,0] = I
	w[5,0] = 0
	W[N-1] = w

	return W
'''
def H_LS():
	W = [0]* N

	for i in range(1,N-1):
		w = np.zeros((6, 6, d, d))
		w[0,0] = I
		w[1,0] = S1
		w[2,0] = S6
		w[3,0] = I
		w[4,0] = I
		w[5,0] = 0
		w[5,1] = S2*-J
		w[5,2] = S7*-J
		w[5,3] = I*Vr
		w[5,4] = I*Vl
		w[5,5] = I
	
		W[i] = w

	w = np.zeros((1, 6, d, d))
	w[0,5] = I   
	W[0] = w

	w = np.zeros((6, 1, d, d))
	w[0,0] = I
	W[N-1] = w
	
	return W
'''
WRS = H_RS()
WLS = H_RS()

################################ Initial LE and RE ###################################

# up = d
# down = d*
# MPO - wL,wR,d,d*

### LE ###

LE = {}

X00 = np.tensordot(M[0][0], WRS[0][:,:,:,[0,2]], axes=(0,3))       #[d] vR       wL wR d [d*]
X00 = np.tensordot(X00[:,:,:,[0,2]], M[0][0], axes=(3,0))        #vR wL wR [d]  [d*] vR*    ==  vR wL wR vR*

X01 = np.tensordot(M[0][0], WRS[0][:,:,:,[0,2]], axes=(0,3))       
X01 = np.tensordot(X01[:,:,:,1:2], M[0][1], axes=(3,0))       

X10 = np.tensordot(M[0][1], WRS[0][:,:,:,1:2], axes=(0,3))      
X10 = np.tensordot(X10[:,:,:,[0,2]], M[0][0], axes=(3,0))       

X11 = np.tensordot(M[0][1], WRS[0][:,:,:,1:2], axes=(0,3))     
X11 = np.tensordot(X11[:,:,:,1:2], M[0][1], axes=(3,0)) 

def LEnv(i,M):
	le=[]
	b0 = [['0 0', '0 2', '2 0', '2 2'], ['0 1', '*', '2 1', '*'], ['1 0', '1 2', '*', '*'], ['1 1', '*', '*', '*']]
	for j in range (4):
		s=0
		for k in range(4):
			if(b0[j][k]!='*'):
				b, c = b0[j][k].split()
				A = np.tensordot(M[i][int(b)], LE[i-1][k], axes=(0,0))              # [vL] d vR  [vR] wL wR vR*                 
				if(b==0):
					A = np.tensordot(A, WRS[i][:,:,:,:1], axes=([0,3],[3,0]))      # [d] vR wL [wR] vR*  [wL] wR d [d*]     
				elif(b==1):
					A = np.tensordot(A, WRS[i][:,:,:,1:2], axes=([0,3],[3,0])) 	
				else:
					A = np.tensordot(A, WRS[i][:,:,:,2:], axes=([0,3],[3,0]))    
				
				if(c==0):
					A = np.tensordot(A[:,:,:,:,:1], M[i][int(c)], axes=([2,4],[0,1]))   # vR wL [vR*] wR [d]  [vL*] [d*] vR* == vR wL wR vR*    
				elif(c==1):
					A = np.tensordot(A[:,:,:,:,1:2], M[i][int(c)], axes=([2,4],[0,1]))
				else:
					A = np.tensordot(A[:,:,:,:,2:], M[i][int(c)], axes=([2,4],[0,1]))
				s=s+A
		le.append(s)	
	return le

LE[0] = [X00,X01,X10,X11]
for i in range(1,6):
	LE[i] = LEnv(i,M)

### RE ###

RE = {}

X00 = np.tensordot(M[N-1][0], WLS[N-1][:,:,:,[0,2]], axes=(1,3))       #vL [d]       wL wR d [d*]
X00 = np.tensordot(X00[:,:,:,[0,2]], M[N-1][0], axes=(3,1))          #vL wL wR [d]  vL* [d*]   == vL wL wR vL*

X01 = np.tensordot(M[N-1][0], WLS[N-1][:,:,:,[0,2]], axes=(1,3))       
X01 = np.tensordot(X01[:,:,:,1:2], M[N-1][1], axes=(3,1))        

X10 = np.tensordot(M[N-1][1], WLS[N-1][:,:,:,1:2], axes=(1,3))      
X10 = np.tensordot(X10[:,:,:,[0,2]], M[N-1][0], axes=(3,1))       

X11 = np.tensordot(M[N-1][1], WLS[N-1][:,:,:,1:2], axes=(1,3))       
X11 = np.tensordot(X11[:,:,:,1:2], M[N-1][1], axes=(3,1)) 

def REnv(i,M):
	re=[]
	b0 = [['0 0', '0 1', '1 0', '1 1'], ['0 2', '*', '1 2', '*'], ['2 0', '2 1', '*', '*'], ['2 2', '*', '*', '*']]
	for j in range (4):
		s=0
		for k in range(4):
			if(b0[j][k]!='*'):
				b, c = b0[j][k].split()   
				A = np.tensordot(M[i][int(b)], RE[i+1][k], axes=(2,0))              # vL d [vR]   [vL] wL wR vL*          
				if(b==0):
					A = np.tensordot(A, WLS[i][:,:,:,:1], axes=([1,2],[3,1]))      # vL [d] [wL] wR vL*   wL [wR] d [d*]    
				elif(b==1):
					A = np.tensordot(A, WLS[i][:,:,:,2:], axes=([1,2],[3,1]))  
				else:
					A = np.tensordot(A, WLS[i][:,:,:,1:2], axes=([1,2],[3,1]))    
				
				if(c==0):
					A = np.tensordot(A[:,:,:,:,:1], M[i][int(c)], axes=([2,4],[2,1]))   # vL wR [vL*] wL [d]   vL* [d*] [vR*]  == vL wR wL vL*  
				elif(c==1):
					A = np.tensordot(A[:,:,:,:,2:], M[i][int(c)], axes=([2,4],[2,1]))

				else:
					A = np.tensordot(A[:,:,:,:,1:2], M[i][int(c)], axes=([2,4],[2,1]))
			
				A = np.swapaxes(A,1,2)   #vL wL wR vL*  
				s=s+A
		re.append(s)	
	return re

RE[N-1] = [X00,X01,X10,X11]
for i in range(N-2,6,-1):    #6
	RE[i] = REnv(i,M)

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

def Lanczos(L, R, gv, wi, wj, td, strg):
	
	def Hv_RS(L, R, GV, wi, wj):
	
		S = 0
		w1=[]
		
		p = np.tensordot(np.tensordot(L[0], GV[0], axes=(0,0)), R[0], axes=(5,0))      #[vR] wL wR vR*  [vL] d d vR = wL wR vR* d d [vR]  [vL] wL wR vL* = wL wR vR* d d wL wR vL*
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))   # wL [wR] vR* [d] d wL wR vL*  [wL] wR d [d*] = wL vR* [d] [wL] wR vL* [wR] d  [wL] [wR] d [d*] = wL vR* wR vL* d d

		p = np.tensordot(np.tensordot(L[0], GV[4], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:2], axes=([1,3],[0,3])), wj[:,:,:,2:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[2], GV[3], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,2:], axes=([1,3],[0,3])), wj[:,:,:,1:2], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[2], GV[2], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,2:], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[1], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,1:2], axes=([2,3,6],[3,1,0]))
	
		S1 = S.copy()
		w1.extend(list(S[0,:,0,:,0,0].flatten()))

		S = 0
	
		p = np.tensordot(np.tensordot(L[0], GV[0], axes=(0,0)), R[1], axes=(5,0))      
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[4], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:2], axes=([1,3],[0,3])), wj[:,:,:,2:], axes=([2,3,6],[3,1,0]))
	
		p = np.tensordot(np.tensordot(L[2], GV[2], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,2:], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[1], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,1:2], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[2], GV[3], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,2:], axes=([1,3],[0,3])), wj[:,:,:,1:2], axes=([2,3,6],[3,1,0]))

		w1.extend(list(S[0,:,0,:,0,1].flatten()))

		S = 0
	
		p = np.tensordot(np.tensordot(L[1], GV[0], axes=(0,0)), R[0], axes=(5,0))      
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0])) 

		p = np.tensordot(np.tensordot(L[1], GV[4], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:2], axes=([1,3],[0,3])), wj[:,:,:,2:], axes=([2,3,6],[3,1,0]))
	
		p = np.tensordot(np.tensordot(L[3], GV[2], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,2:], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[1], GV[1], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,1:2], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[3], GV[3], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,2:], axes=([1,3],[0,3])), wj[:,:,:,1:2], axes=([2,3,6],[3,1,0]))

		w1.extend(list(S[0,:,0,:,2,0].flatten()))

		S = 0
	
		p = np.tensordot(np.tensordot(L[1], GV[0], axes=(0,0)), R[1], axes=(5,0))      
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0])) 

		p = np.tensordot(np.tensordot(L[1], GV[4], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:2], axes=([1,3],[0,3])), wj[:,:,:,2:], axes=([2,3,6],[3,1,0]))
	
		p = np.tensordot(np.tensordot(L[3], GV[2], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,2:], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[1], GV[1], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,1:2], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[3], GV[3], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,2:], axes=([1,3],[0,3])), wj[:,:,:,1:2], axes=([2,3,6],[3,1,0]))

		w1.extend(list(S[0,:,0,:,2,1].flatten()))
		w1.extend(list(S1[0,:,0,:,1,2].flatten()))
		
		return w1

	def Hv_LS(L, R, GV, wi, wj):
	
		S = 0
		w1=[]
		
		p = np.tensordot(np.tensordot(L[0], GV[0], axes=(0,0)), R[0], axes=(5,0))   
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0])) 

		p = np.tensordot(np.tensordot(L[0], GV[4], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,2:], axes=([1,3],[0,3])), wj[:,:,:,1:2], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[2], GV[3], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:2], axes=([1,3],[0,3])), wj[:,:,:,2:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[2], GV[2], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:2], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[1], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,2:], axes=([2,3,6],[3,1,0]))
	
		S1 = S.copy()
		w1.extend(list(S[0,:,0,:,0,0].flatten()))

		S = 0
	
		p = np.tensordot(np.tensordot(L[0], GV[0], axes=(0,0)), R[1], axes=(5,0))      
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[4], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,2:], axes=([1,3],[0,3])), wj[:,:,:,1:2], axes=([2,3,6],[3,1,0]))
	
		p = np.tensordot(np.tensordot(L[2], GV[2], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:2], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[0], GV[1], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,2:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[2], GV[3], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:2], axes=([1,3],[0,3])), wj[:,:,:,2:], axes=([2,3,6],[3,1,0]))

		w1.extend(list(S[0,:,0,:,0,2].flatten()))

		S = 0
	
		p = np.tensordot(np.tensordot(L[1], GV[0], axes=(0,0)), R[0], axes=(5,0))      
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0])) 

		p = np.tensordot(np.tensordot(L[1], GV[4], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,2:], axes=([1,3],[0,3])), wj[:,:,:,1:2], axes=([2,3,6],[3,1,0]))
	
		p = np.tensordot(np.tensordot(L[3], GV[2], axes=(0,0)), R[0], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:2], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[1], GV[1], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,2:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[3], GV[3], axes=(0,0)), R[2], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:2], axes=([1,3],[0,3])), wj[:,:,:,2:], axes=([2,3,6],[3,1,0]))

		w1.extend(list(S[0,:,0,:,1,0].flatten()))

		S = 0
	
		p = np.tensordot(np.tensordot(L[1], GV[0], axes=(0,0)), R[1], axes=(5,0))      
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0])) 

		p = np.tensordot(np.tensordot(L[1], GV[4], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,2:], axes=([1,3],[0,3])), wj[:,:,:,1:2], axes=([2,3,6],[3,1,0]))
	
		p = np.tensordot(np.tensordot(L[3], GV[2], axes=(0,0)), R[1], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:2], axes=([1,3],[0,3])), wj[:,:,:,:1], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[1], GV[1], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,:1], axes=([1,3],[0,3])), wj[:,:,:,2:], axes=([2,3,6],[3,1,0]))

		p = np.tensordot(np.tensordot(L[3], GV[3], axes=(0,0)), R[3], axes=(5,0))     
		S = S + np.tensordot(np.tensordot(p, wi[:,:,:,1:2], axes=([1,3],[0,3])), wj[:,:,:,2:], axes=([2,3,6],[3,1,0]))

		w1.extend(list(S[0,:,0,:,1,2].flatten()))
		w1.extend(list(S1[0,:,0,:,2,1].flatten()))
		
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

	if(strg == 'RS'):
		w = Hv_RS(L, R, gv, wi, wj)
	if(strg == 'LS'):
		w = Hv_LS(L, R, gv, wi, wj)

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

		if(strg == 'RS'):
			w = Hv_RS(L, R, GV, wi, wj)
		if(strg == 'LS'):
			w = Hv_LS(L, R, GV, wi, wj)

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
	print(chi)
	if(chi > td):
		chi_ = td
	else:
		chi_ = chi
	
	chi_ = len(np.hstack((S0,S1)))
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

	print(len(s0),len(s1))

	s0.extend(s1)
	S = np.diag(S) #s0)

	return (E,[mi00, mi01, mi10],[mj00, mj01, mj10],S)


##################################### Sweeps ####################################################

def sweeps(n):
	
	c=0
	Eng = []
	iters = []

	SS = [ 0,0,0,0,0, np.diag([1/np.sqrt(d0+d1)]*(d0+d1)) ]    #13
	SS.extend([ 0 ]*(N-7))
	td = 100
	for i in range(n):
		for j in range(6,N-6):      
			c=c+1
			gv = guess_vector_RS(j,SS[j-1],M)
			E, M[j], M[j+1], SS[j] = Lanczos(LE[j-1], RE[j+2], gv, WRS[j], WRS[j+1],td,'RS')
	
			LE[j] = LEnv(j,M)
			RE[j+1] = REnv(j+1,M)
			
			Eng.append(E)
			iters.append(c)
			
			print("Sweep #%i, RS, site%i"%(i,j), E, '\n')

		td=td+100
		for j in range(N-8,4,-1):         

			c=c+1
			gv = guess_vector_LS(j,SS[j+1],M)
			E, M[j], M[j+1], SS[j] = Lanczos(LE[j-1], RE[j+2], gv, WLS[j], WLS[j+1],td,'LS')
		
			LE[j] = LEnv(j,M)
			RE[j+1] = REnv(j+1,M)
			
			Eng.append(E)
			iters.append(c)

			print("Sweep #%i, LS, site%i"%(i,j), E, '\n')
		td=td+100

	return Eng,iters


##################################### Run ########################################################
st = time.time()
Eng, iters = sweeps(5)
plt.plot(iters[:],Eng[:])
plt.show()
#plt.plot(iters[17:],Eng[17:])
#plt.show()
#print(time.time()-st)

erew

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
	

	
