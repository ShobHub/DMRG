import matplotlib.pylab as plt
from termcolor import colored
import itertools as it
from collections import deque
from Lanczos_Hv import Lanc_Hv as LHv
from DimerDensity import dimerDen as DD
from scipy.optimize import curve_fit
import numpy as np
import pickle
import time

class DMRG_AB():
	def __init__(self,J,Vr,Vl,Vlr,ladN,lm):
		self.J = J
		self.Vr = Vr
		self.Vl = Vl
		self.Vlr = Vlr
		self.ladN = ladN
		self.lm = lm

	def exact_states(self,lad):
		M={}
		# Right Normalised - Exact
		d0 = 1
		d1 = 1
		N = len(lad)
		B0 = np.zeros([d0,1])
		B1 = np.zeros([d1,1])
		B0[0,0] = 1 
		B1[0,0] = 1
		M[N-1] = [B0, B1] 
		for i in range(N-2,N-6,-1):
			if(lad[i] == 's'):
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
			elif(lad[i] == 'A'):
				I = np.eye(2*d0+d1)
				B0r10 = np.zeros([2*d0+d1,1,d0])
				B0r30 = np.zeros([2*d0+d1,1,d0])
				B0r21 = np.zeros([2*d0+d1,1,d1])
				B1r40 = np.zeros([d0, 1, d0])

				B0r10[:,0,:] = I[:,:d0]
				B0r30[:,0,:] = I[:,d0:2*d0]
				B0r21[:,0,:] = I[:,2*d0:2*d0+d1]
				B1r40[:,0,:] = np.eye(d0)
				
				M[i] = [B0r10, B0r30, B0r21, B1r40]
				dt = 2*d0+d1
				d1 = d0
				d0 = dt
			else:
				I = np.eye(3*d0+d1)
				B0h20 = np.zeros([3*d0+d1,1,d0])
				B0h30 = np.zeros([3*d0+d1,1,d0])
				B0h40 = np.zeros([3*d0+d1,1,d0])
				B0h51 = np.zeros([3*d0+d1,1,d1])
				B1h10 = np.zeros([d0, 1, d0])

				B0h20[:,0,:] = I[:,:d0]
				B0h30[:,0,:] = I[:,d0:2*d0]
				B0h40[:,0,:] = I[:,2*d0:3*d0]
				B0h51[:,0,:] = I[:,3*d0:3*d0+d1]
				B1h10[:,0,:] = np.eye(d0)
				
				M[i] = [B0h20, B0h30, B0h40, B0h51, B1h10]
				dt = 3*d0+d1
				d1 = d0
				d0 = dt
			
		# Right Normalised - Random
		for i in range(6,N-5):
			if(lad[i] == 's'):
				a1 = np.random.uniform(-1,1,(d0,d0+d1))       
				a2 = np.random.uniform(-1,1,(d1,d0))       
	
				_,_,V1 = np.linalg.svd(a1, full_matrices=False)
				_,_,M3 = np.linalg.svd(a2, full_matrices=False)

				B00 = np.zeros([d0,1,d0])     
				B01 = np.zeros([d0,1,d1])      
				B10 = np.zeros([d1,1,d0])   
	
				B00[:,0,:]=V1[:,:d0]       
				B01[:,0,:]=V1[:,d0:]       
				B10[:,0,:]=M3
	
				M[i] = [B00, B01, B10] 
			elif(lad[i] == 'A'):
				a1 = np.random.uniform(-1,1,(d0,2*d0+d1))       
				a2 = np.random.uniform(-1,1,(d1,d0)) 
			
				_,_,V1 = np.linalg.svd(a1, full_matrices=False)
				_,_,M3 = np.linalg.svd(a2, full_matrices=False)

				B0r10 = np.zeros([d0,1,d0])
				B0r30 = np.zeros([d0,1,d0])
				B0r21 = np.zeros([d0,1,d1])
				B1r40 = np.zeros([d1,1,d0])

				B0r10[:,0,:] = V1[:,:d0]
				B0r30[:,0,:] = V1[:,d0:2*d0]
				B0r21[:,0,:] = V1[:,2*d0:2*d0+d1]
				B1r40[:,0,:] = M3
				
				M[i] = [B0r10, B0r30, B0r21, B1r40]
			else:
				a1 = np.random.uniform(-1,1,(d0,3*d0+d1))       
				a2 = np.random.uniform(-1,1,(d1,d0)) 
			
				_,_,V1 = np.linalg.svd(a1, full_matrices=False)
				_,_,M3 = np.linalg.svd(a2, full_matrices=False)

				B0h20 = np.zeros([d0,1,d0])
				B0h30 = np.zeros([d0,1,d0])
				B0h40 = np.zeros([d0,1,d0])
				B0h51 = np.zeros([d0,1,d1])
				B1h10 = np.zeros([d1,1,d0])

				B0h20[:,0,:] = V1[:,:d0]
				B0h30[:,0,:] = V1[:,d0:2*d0]
				B0h40[:,0,:] = V1[:,2*d0:3*d0]
				B0h51[:,0,:] = V1[:,3*d0:3*d0+d1]
				B1h10[:,0,:] = M3
				
				M[i] = [B0h20, B0h30, B0h40, B0h51, B1h10]
		d0_r = d0
		d1_r = d1

		# Left Normalised - Exact
		d0 = 1
		d1 = 1
		B0 = np.zeros([1,d0])
		B1 = np.zeros([1,d1])
		B0[0,0] = 1 
		B1[0,0] = 1
		M[0] = [B0, B1] 
		for i in range(1,5):
			if(lad[i] == 's'):
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
			elif(lad[i] == 'A'):
				I = np.eye(2*d0+d1)
				B0r10 = np.zeros([d0, 1, 2*d0+d1])
				B0r30 = np.zeros([d0, 1, 2*d0+d1])
				B1r40 = np.zeros([d1, 1, 2*d0+d1])
				B0r21 = np.zeros([d0, 1, d0])

				B0r10[:,0,:] = I[:d0,:]
				B0r30[:,0,:] = I[d0:2*d0,:]
				B1r40[:,0,:] = I[2*d0:2*d0+d1,:]
				B0r21[:,0,:] = np.eye(d0)
				
				M[i] = [B0r10, B0r30, B0r21, B1r40]
				dt = 2*d0+d1
				d1 = d0
				d0 = dt
			else:
				I = np.eye(3*d0+d1)
				B0h20 = np.zeros([d0, 1, 3*d0+d1])
				B0h30 = np.zeros([d0, 1, 3*d0+d1])
				B0h40 = np.zeros([d0, 1, 3*d0+d1])
				B1h10 = np.zeros([d1, 1, 3*d0+d1])
				B0h51 = np.zeros([d0, 1, d0])

				B0h20[:,0,:] = I[:d0,:]
				B0h30[:,0,:] = I[d0:2*d0,:]
				B0h40[:,0,:] = I[2*d0:3*d0,:]
				B1h10[:,0,:] = I[3*d0:3*d0+d1,:]
				B0h51[:,0,:] = np.eye(d0)
				
				M[i] = [B0h20, B0h30, B0h40, B0h51, B1h10]
				dt = 3*d0+d1
				d1 = d0
				d0 = dt

		# Left Normalised - Random
		if(lad[5] == 's'):
			a1 = np.random.uniform(-1,1,(d0+d1,d0_r))   
			a2 = np.random.uniform(-1,1,(d0,d1_r))     
	
			V1,_,_ = np.linalg.svd(a1, full_matrices=False)
			M3,_,_ = np.linalg.svd(a2, full_matrices=False)

			B00 = np.zeros([d0,1,d0_r])     
			B01 = np.zeros([d0,1,d1_r])     
			B10 = np.zeros([d1,1,d0_r])      

			B00[:,0,:]=V1[:d0,:]        
			B01[:,0,:]=M3
			B10[:,0,:]=V1[d0:,:]        

			M[5] = [B00, B01, B10] 
		elif(lad[5] == 'A'):
			a1 = np.random.uniform(-1,1,(2*d0+d1,d0_r))   
			a2 = np.random.uniform(-1,1,(d0,d1_r))     
	
			V1,_,_ = np.linalg.svd(a1, full_matrices=False)
			M3,_,_ = np.linalg.svd(a2, full_matrices=False)

			B0r10 = np.zeros([d0, 1, d0_r])
			B0r30 = np.zeros([d0, 1, d0_r])
			B1r40 = np.zeros([d1, 1, d0_r])
			B0r21 = np.zeros([d0, 1, d1_r])

			B0r10[:,0,:] = V1[:d0,:]
			B0r30[:,0,:] = V1[d0:2*d0,:]
			B1r40[:,0,:] = V1[2*d0:2*d0+d1,:]
			B0r21[:,0,:] = M3

			M[5] = [B0r10, B0r30, B0r21, B1r40]
		else:
			a1 = np.random.uniform(-1,1,(3*d0+d1,d0_r))   
			a2 = np.random.uniform(-1,1,(d0,d1_r))     
	
			V1,_,_ = np.linalg.svd(a1, full_matrices=False)
			M3,_,_ = np.linalg.svd(a2, full_matrices=False)

			B0h20 = np.zeros([d0, 1, d0_r])
			B0h30 = np.zeros([d0, 1, d0_r])
			B0h40 = np.zeros([d0, 1, d0_r])
			B1h10 = np.zeros([d1, 1, d0_r])
			B0h51 = np.zeros([d0, 1, d1_r])

			B0h20[:,0,:] = V1[:d0,:]
			B0h30[:,0,:] = V1[d0:2*d0,:]
			B0h40[:,0,:] = V1[2*d0:3*d0,:]
			B1h10[:,0,:] = V1[3*d0:3*d0+d1,:]
			B0h51[:,0,:] = M3
			
			M[5] = [B0h20, B0h30, B0h40, B0h51, B1h10]

		return M,d0_r,d1_r

	# Matrix Product Operator 
	def MPO(self, lad):	
		Sp = np.array([[0., 1.], [0., 0.]])
		Sm = np.array([[0., 0.], [1., 0.]])

		Sr_12 = np.zeros((4,4))
		Sr_12[1][0] = 1
		Sr_21 = np.zeros((4,4))
		Sr_21[0][1] = 1
		Sr_31 = np.zeros((4,4))
		Sr_31[0][2] = 1
		Sr_13 = np.zeros((4,4))
		Sr_13[2][0] = 1
		Sr_34 = np.zeros((4,4))
		Sr_34[3][2] = 1
		Sr_43 = np.zeros((4,4))
		Sr_43[2][3] = 1

		Srr_32 = np.zeros((5,5))
		Srr_32[1][2] = 1
		Srr_23 = np.zeros((5,5))
		Srr_23[2][1] = 1
		Srr_42 = np.zeros((5,5))
		Srr_42[1][3] = 1
		Srr_24 = np.zeros((5,5))
		Srr_24[3][1] = 1
		Srr_31 = np.zeros((5,5))
		Srr_31[0][2] = 1
		Srr_13 = np.zeros((5,5))
		Srr_13[2][0] = 1
		Srr_45 = np.zeros((5,5))
		Srr_45[4][3] = 1
		Srr_54 = np.zeros((5,5))
		Srr_54[3][4] = 1
		I2 = np.eye(2)
		I4 = np.eye(4)
		I5 = np.eye(5)
		n = np.matmul(Sp,Sm)

		Nr =[]
		for i in range(4):	
			nr = np.zeros((4,4))
			nr[i][i] = 1
			Nr.append(nr)
		Nrr =[]
		for i in range(5):	
			nrr = np.zeros((5,5))
			nrr[i][i] = 1
			Nrr.append(nrr)

		N = len(lad)
		W = [0]*N
		
		w = np.zeros((1, 6, 2, 2))
		w[0,1] = -self.J*Sm
		w[0,2] = -self.J*Sp
		w[0,3] = self.Vl*(I2-n)
		w[0,4] = self.Vr*(n)
		w[0,5] = I2
		W[0] = w

		w = np.zeros((6, 1, 2, 2))
		w[0,0] = I2
		w[1,0] = Sm
		w[2,0] = Sp
		w[3,0] = (I2-n)
		w[4,0] = (n)
		W[N-1] = w

		ws = np.zeros((6, 6, 2, 2))
		ws[0, 0] = I2
		ws[1, 0] = Sm
		ws[2, 0] = Sp
		ws[3, 0] = I2 - n
		ws[4, 0] = n
		ws[5, 1] = -self.J * Sm
		ws[5, 2] = -self.J * Sp
		ws[5, 3] = self.Vl * (I2 - n)
		ws[5, 4] = self.Vr * n
		ws[5, 5] = I2

		wA = np.zeros((6, 6, 4, 4))
		wA[0, 0] = I4
		wA[1, 0] = Sr_34
		wA[2, 0] = Sr_43
		wA[3, 0] = Nr[3]
		wA[4, 0] = Nr[2]
		wA[5, 0] = -self.J * (Sr_31 + Sr_13) + self.Vl * (Nr[1] + Nr[3]) + self.Vlr * (Nr[0] + Nr[2])
		wA[5, 1] = -self.J * Sr_12
		wA[5, 2] = -self.J * Sr_21
		wA[5, 3] = self.Vl * Nr[1]
		wA[5, 4] = self.Vr * Nr[0]
		wA[5, 5] = I4

		wB = np.zeros((6, 6, 5, 5))
		wB[0, 0] = I5
		wB[1, 0] = Srr_31
		wB[2, 0] = Srr_13
		wB[3, 0] = Nrr[0]
		wB[4, 0] = Nrr[2]
		wB[5, 0] = -self.J * (Srr_32 + Srr_23 + Srr_42 + Srr_24) + self.Vl * (Nrr[0] + Nrr[4]) + self.Vlr * (Nrr[1] + Nrr[2] + Nrr[3])
		wB[5, 1] = -self.J * Srr_45
		wB[5, 2] = -self.J * Srr_54
		wB[5, 3] = self.Vl * Nrr[4]
		wB[5, 4] = self.Vr * Nrr[3]
		wB[5, 5] = I5

		for i in range(1, N - 1):
			if (lad[i] == 'A'):
				W[i] = wA
			elif (lad[i] == 'B'):
				W[i] = wB
			elif (lad[i] == 's'):
				W[i] = ws
			else:
				print(i, lad[i - 1:i + 2])
		return W

	# Building Left Environment on site `s'
	def LEnv_s(self,i,M,LE,W):    
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

	# Building Left Environment on site `A'
	#B0r10, B0r30, B0r21, B1r40
	def LEnv_A(self,i,M,LE,W):
		le=[]
		b0 = [[['0 0','0 1','1 0','1 1'],['0 3','1 3'],['3 0','3 1'],['3 3']], [['0 2','1 2'],['*'],['3 2'],['*']], [['2 0', '2 1'],['2 3'],['*'],['*']], [['2 2'],['*'],['*'],['*']]]
		for j in range(4):
			s=0
			for k in range(4):
				for p in range(len(b0[j][k])):
					if(b0[j][k][p]!='*'):
						b, c = b0[j][k][p].split()
						A = np.tensordot(M[i][int(b)], LE[i-1][k], axes=(0,0))              # [vL] d vR  [vR] wL wR vR*                 
						if(b=='0'):
							A = np.tensordot(A, W[i][:,:,:,:1], axes=([0,3],[3,0]))      # [d] vR wL [wR] vR*  [wL] wR d [d*]
						elif(b=='1'):     
							A = np.tensordot(A, W[i][:,:,:,2:3], axes=([0,3],[3,0]))
						elif(b=='2'):     
							A = np.tensordot(A, W[i][:,:,:,1:2], axes=([0,3],[3,0]))
						else:		
							A = np.tensordot(A, W[i][:,:,:,3:], axes=([0,3],[3,0]))  
						if(c=='0'):
							A = np.tensordot(A[:,:,:,:,:1], M[i][int(c)], axes=([2,4],[0,1]))   # vR wL [vR*] wR [d]  [vL*] [d*] vR* == vR wL wR vR* 
						elif(c=='1'):
							A = np.tensordot(A[:,:,:,:,2:3], M[i][int(c)], axes=([2,4],[0,1])) 
						elif(c=='2'):
							A = np.tensordot(A[:,:,:,:,1:2], M[i][int(c)], axes=([2,4],[0,1]))    
						else:
							A = np.tensordot(A[:,:,:,:,3:], M[i][int(c)], axes=([2,4],[0,1]))
						s=s+A
			le.append(s)	
		return le

	# Building Left Environment on site `B'
	#B0h20, B0h30, B0h40, B0h51, B1h10
	def LEnv_B(self,i,M,LE,W):
		le=[]
		b0 = [[['0 0','0 1','0 2','1 0','1 1','1 2','2 0','2 1','2 2'],['0 4','1 4','2 4'],['4 0','4 1','4 2'],['4 4']], [['0 3','1 3','2 3'],['*'],['4 3'],['*']], [['3 0','3 1','3 2'],['3 4'],['*'],['*']], [['3 3'],['*'],['*'],['*']]]
		for j in range(4):
			s=0
			for k in range(4):
				for p in range(len(b0[j][k])):
					if(b0[j][k][p]!='*'):
						b, c = b0[j][k][p].split()
						A = np.tensordot(M[i][int(b)], LE[i-1][k], axes=(0,0))              # [vL] d vR  [vR] wL wR vR*                 
						if(b=='0'):
							A = np.tensordot(A, W[i][:,:,:,1:2], axes=([0,3],[3,0]))      # [d] vR wL [wR] vR*  [wL] wR d [d*]
						elif(b=='1'):     
							A = np.tensordot(A, W[i][:,:,:,2:3], axes=([0,3],[3,0]))
						elif(b=='2'):     
							A = np.tensordot(A, W[i][:,:,:,3:4], axes=([0,3],[3,0]))
						elif(b=='3'):     
							A = np.tensordot(A, W[i][:,:,:,4:], axes=([0,3],[3,0]))
						else:		
							A = np.tensordot(A, W[i][:,:,:,:1], axes=([0,3],[3,0]))  
						if(c=='0'):
							A = np.tensordot(A[:,:,:,:,1:2], M[i][int(c)], axes=([2,4],[0,1]))   # vR wL [vR*] wR [d]  [vL*] [d*] vR* == vR wL wR vR* 
						elif(c=='1'):
							A = np.tensordot(A[:,:,:,:,2:3], M[i][int(c)], axes=([2,4],[0,1])) 
						elif(c=='2'):
							A = np.tensordot(A[:,:,:,:,3:4], M[i][int(c)], axes=([2,4],[0,1]))  
						elif(c=='3'):
							A = np.tensordot(A[:,:,:,:,4:], M[i][int(c)], axes=([2,4],[0,1]))    
						else:
							A = np.tensordot(A[:,:,:,:,:1], M[i][int(c)], axes=([2,4],[0,1]))
						s=s+A
			le.append(s)	
		return le

	# Building Right Environment on site `s'
	def REnv_s(self,i,M,RE,W):
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
		
	# Building Right Environment on site `A'
	#B0r10, B0r30, B0r21, B1r40
	def REnv_A(self,i,M,RE,W):
		re=[]
		b0 = [[['0 0','0 1','1 0','1 1'],['0 2','1 2'],['2 0','2 1'],['2 2']], [['0 3','1 3'],['*'],['2 3'],['*']], [['3 0', '3 1'],['3 2'],['*'],['*']], [['3 3'],['*'],['*'],['*']]]
		for j in range (4):
			s=0
			for k in range(4):
				for p in range(len(b0[j][k])):
					if(b0[j][k][p]!='*'):
						b, c = b0[j][k][p].split()   
						A = np.tensordot(M[i][int(b)], RE[i+1][k], axes=(2,0))              # vL d [vR]   [vL] wL wR vL*          
						if(b=='0'):
							A = np.tensordot(A, W[i][:,:,:,:1], axes=([1,2],[3,1]))      # vL [d] [wL] wR vL*   wL [wR] d [d*] 
						elif(b=='1'):
							A = np.tensordot(A, W[i][:,:,:,2:3], axes=([1,2],[3,1])) 
						elif(b=='2'):
							A = np.tensordot(A, W[i][:,:,:,1:2], axes=([1,2],[3,1]))      
						else:
							A = np.tensordot(A, W[i][:,:,:,3:], axes=([1,2],[3,1]))    
						if(c=='0'):
							A = np.tensordot(A[:,:,:,:,:1], M[i][int(c)], axes=([2,4],[2,1]))   # vL wR [vL*] wL [d]   vL* [d*] [vR*]  == vL wR wL vL*  
						elif(c=='1'):
							A = np.tensordot(A[:,:,:,:,2:3], M[i][int(c)], axes=([2,4],[2,1])) 
						elif(c=='2'):
							A = np.tensordot(A[:,:,:,:,1:2], M[i][int(c)], axes=([2,4],[2,1])) 
						else:
							A = np.tensordot(A[:,:,:,:,3:], M[i][int(c)], axes=([2,4],[2,1]))
						A = np.swapaxes(A,1,2)   #vL wL wR vL*  
						s=s+A
			re.append(s)	
		return re

	# Building Right Environment on site `B'
	#B0h20, B0h30, B0h40, B0h51, B1h10
	def REnv_B(self,i,M,RE,W):
		re=[]
		b0 = [[['0 0','0 1','0 2','1 0','1 1','1 2','2 0','2 1','2 2'],['0 3','1 3','2 3'],['3 0','3 1','3 2'],['3 3']], [['0 4','1 4','2 4'],['*'],['3 4'],['*']], [['4 0','4 1','4 2'],['4 3'],['*'],['*']], [['4 4'],['*'],['*'],['*']]]
		for j in range (4):
			s=0
			for k in range(4):
				for p in range(len(b0[j][k])):
					if(b0[j][k][p]!='*'):
						b, c = b0[j][k][p].split()   
						A = np.tensordot(M[i][int(b)], RE[i+1][k], axes=(2,0))              # vL d [vR]   [vL] wL wR vL*          
						if(b=='0'):
							A = np.tensordot(A, W[i][:,:,:,1:2], axes=([1,2],[3,1]))      # vL [d] [wL] wR vL*   wL [wR] d [d*] 
						elif(b=='1'):
							A = np.tensordot(A, W[i][:,:,:,2:3], axes=([1,2],[3,1])) 
						elif(b=='2'):
							A = np.tensordot(A, W[i][:,:,:,3:4], axes=([1,2],[3,1]))    
						elif(b=='3'):
							A = np.tensordot(A, W[i][:,:,:,4:], axes=([1,2],[3,1]))      
						else:
							A = np.tensordot(A, W[i][:,:,:,:1], axes=([1,2],[3,1]))    
						if(c=='0'):
							A = np.tensordot(A[:,:,:,:,1:2], M[i][int(c)], axes=([2,4],[2,1]))   # vL wR [vL*] wL [d]   vL* [d*] [vR*]  == vL wR wL vL*  
						elif(c=='1'):
							A = np.tensordot(A[:,:,:,:,2:3], M[i][int(c)], axes=([2,4],[2,1])) 
						elif(c=='2'):
							A = np.tensordot(A[:,:,:,:,3:4], M[i][int(c)], axes=([2,4],[2,1])) 
						elif(c=='3'):
							A = np.tensordot(A[:,:,:,:,4:], M[i][int(c)], axes=([2,4],[2,1])) 
						else:
							A = np.tensordot(A[:,:,:,:,:1], M[i][int(c)], axes=([2,4],[2,1]))
						A = np.swapaxes(A,1,2)   #vL wL wR vL*  
						s=s+A
			re.append(s)	
		return re
	
	def Envs(self,lad,M,W):
		N = len(lad)
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

		for i in range(N-2,6,-1):
			if(lad[i] == 's'):
				RE[i] = self.REnv_s(i,M,RE,W) 
			elif(lad[i] == 'A'):
				RE[i] = self.REnv_A(i,M,RE,W) 
			elif(lad[i] == 'B'):
				RE[i] = self.REnv_B(i,M,RE,W) 
		for i in range(1,6):
			if(lad[i] == 's'):
				LE[i] = self.LEnv_s(i,M,LE,W) 
			elif(lad[i] == 'A'):
				LE[i] = self.LEnv_A(i,M,LE,W) 
			elif(lad[i] == 'B'):
				LE[i] = self.LEnv_B(i,M,LE,W) 
		return LE,RE

	# Two-site Initialisation
	def guess_vector_RS(self,i,ss,M):
		r0 = M[i][0].shape[0]
		l0 = M[i-1][0].shape[2]
		B0rr0 = np.tensordot(np.tensordot(ss[:l0,:r0], M[i][0], axes=(1,0)), M[i+1][0], axes=(2,0))    # vL d [vR] [vL] d vR = vL d d vR
		B0re1 = np.tensordot(np.tensordot(ss[:l0,:r0], M[i][0], axes=(1,0)), M[i+1][1], axes=(2,0))
		B1er0 = np.tensordot(np.tensordot(ss[l0:,r0:], M[i][2], axes=(1,0)), M[i+1][0], axes=(2,0))
		B1ee1 = np.tensordot(np.tensordot(ss[l0:,r0:], M[i][2], axes=(1,0)), M[i+1][1], axes=(2,0))
		B0ee0 = np.tensordot(np.tensordot(ss[:l0,:r0], M[i][1], axes=(1,0)), M[i+1][2], axes=(2,0))
	
		return [B0rr0, B0re1, B1er0, B1ee1, B0ee0]
	def guess_vector_RS_sA(self,i,ss,M):
		r0 = M[i][0].shape[0]
		l0 = M[i-1][0].shape[2]
		B0rr10 = np.tensordot(np.tensordot(ss[:l0,:r0], M[i][0], axes=(1,0)), M[i+1][0], axes=(2,0))    # vL d [vR] [vL] d vR = vL d d vR
		B0rr30 = np.tensordot(np.tensordot(ss[:l0,:r0], M[i][0], axes=(1,0)), M[i+1][1], axes=(2,0))
		B0rr21 = np.tensordot(np.tensordot(ss[:l0,:r0], M[i][0], axes=(1,0)), M[i+1][2], axes=(2,0))
		B1er10 = np.tensordot(np.tensordot(ss[l0:,r0:], M[i][2], axes=(1,0)), M[i+1][0], axes=(2,0))
		B1er30 = np.tensordot(np.tensordot(ss[l0:,r0:], M[i][2], axes=(1,0)), M[i+1][1], axes=(2,0))
		B1er21 = np.tensordot(np.tensordot(ss[l0:,r0:], M[i][2], axes=(1,0)), M[i+1][2], axes=(2,0))
		B0er40 = np.tensordot(np.tensordot(ss[:l0,:r0], M[i][1], axes=(1,0)), M[i+1][3], axes=(2,0))
	
		return [B0rr10,B0rr30,B0rr21,B1er10,B1er30,B1er21,B0er40]
	def guess_vector_RS_As(self,i,ss,M):
		r0 = M[i][0].shape[0]
		l0 = M[i-1][0].shape[2]
		B0r1r0 = np.tensordot(np.tensordot(ss[:l0,:r0], M[i][0], axes=(1,0)), M[i+1][0], axes=(2,0))    # vL d [vR] [vL] d vR = vL d d vR
		B0r1e1 = np.tensordot(np.tensordot(ss[:l0,:r0], M[i][0], axes=(1,0)), M[i+1][1], axes=(2,0))
		B0r3r0 = np.tensordot(np.tensordot(ss[:l0,:r0], M[i][1], axes=(1,0)), M[i+1][0], axes=(2,0))
		B0r3e1 = np.tensordot(np.tensordot(ss[:l0,:r0], M[i][1], axes=(1,0)), M[i+1][1], axes=(2,0))
		B1r4r0 = np.tensordot(np.tensordot(ss[l0:,r0:], M[i][3], axes=(1,0)), M[i+1][0], axes=(2,0))
		B1r4e1 = np.tensordot(np.tensordot(ss[l0:,r0:], M[i][3], axes=(1,0)), M[i+1][1], axes=(2,0))
		B0r2e0 = np.tensordot(np.tensordot(ss[:l0,:r0], M[i][2], axes=(1,0)), M[i+1][2], axes=(2,0))
	
		return [B0r1r0,B0r1e1,B0r3r0,B0r3e1,B1r4r0,B1r4e1,B0r2e0]
	def guess_vector_RS_sB(self,i,ss,M):
		r0 = M[i][0].shape[0]
		l0 = M[i-1][0].shape[2]
		B0rh20 = np.tensordot(np.tensordot(ss[:l0,:r0], M[i][0], axes=(1,0)), M[i+1][0], axes=(2,0))    # vL d [vR] [vL] d vR = vL d d vR
		B0rh30 = np.tensordot(np.tensordot(ss[:l0,:r0], M[i][0], axes=(1,0)), M[i+1][1], axes=(2,0))
		B0rh40 = np.tensordot(np.tensordot(ss[:l0,:r0], M[i][0], axes=(1,0)), M[i+1][2], axes=(2,0))
		B0rh51 = np.tensordot(np.tensordot(ss[:l0,:r0], M[i][0], axes=(1,0)), M[i+1][3], axes=(2,0))
		B1eh20 = np.tensordot(np.tensordot(ss[l0:,r0:], M[i][2], axes=(1,0)), M[i+1][0], axes=(2,0))
		B1eh30 = np.tensordot(np.tensordot(ss[l0:,r0:], M[i][2], axes=(1,0)), M[i+1][1], axes=(2,0))
		B1eh40 = np.tensordot(np.tensordot(ss[l0:,r0:], M[i][2], axes=(1,0)), M[i+1][2], axes=(2,0))
		B1eh51 = np.tensordot(np.tensordot(ss[l0:,r0:], M[i][2], axes=(1,0)), M[i+1][3], axes=(2,0))
		B0eh10 = np.tensordot(np.tensordot(ss[:l0,:r0], M[i][1], axes=(1,0)), M[i+1][4], axes=(2,0))
	
		return [B0rh20,B0rh30,B0rh40,B0rh51,B1eh20,B1eh30,B1eh40,B1eh51,B0eh10]
	def guess_vector_RS_Bs(self,i,ss,M):
		r0 = M[i][0].shape[0]
		l0 = M[i-1][0].shape[2]
		B0h2r0 = np.tensordot(np.tensordot(ss[:l0,:r0], M[i][0], axes=(1,0)), M[i+1][0], axes=(2,0))    # vL d [vR] [vL] d vR = vL d d vR
		B0h2e1 = np.tensordot(np.tensordot(ss[:l0,:r0], M[i][0], axes=(1,0)), M[i+1][1], axes=(2,0))
		B0h3r0 = np.tensordot(np.tensordot(ss[:l0,:r0], M[i][1], axes=(1,0)), M[i+1][0], axes=(2,0))
		B0h3e1 = np.tensordot(np.tensordot(ss[:l0,:r0], M[i][1], axes=(1,0)), M[i+1][1], axes=(2,0))
		B0h4r0 = np.tensordot(np.tensordot(ss[:l0,:r0], M[i][2], axes=(1,0)), M[i+1][0], axes=(2,0))
		B0h4e1 = np.tensordot(np.tensordot(ss[:l0,:r0], M[i][2], axes=(1,0)), M[i+1][1], axes=(2,0))
		B1h1r0 = np.tensordot(np.tensordot(ss[l0:,r0:], M[i][4], axes=(1,0)), M[i+1][0], axes=(2,0))
		B1h1e1 = np.tensordot(np.tensordot(ss[l0:,r0:], M[i][4], axes=(1,0)), M[i+1][1], axes=(2,0))
		B0h5e0 = np.tensordot(np.tensordot(ss[:l0,:r0], M[i][3], axes=(1,0)), M[i+1][2], axes=(2,0))
	
		return [B0h2r0,B0h2e1,B0h3r0,B0h3e1,B0h4r0,B0h4e1,B1h1r0,B1h1e1,B0h5e0]
	def guess_vector_LS(self,i,ss,M):
		l0 = M[i+1][0].shape[2]
		r0 = M[i+2][0].shape[0]
		B0rr0 = np.tensordot(np.tensordot(M[i][0], M[i+1][0], axes=(2,0)), ss[:l0,:r0], axes=(3,0))    # vL d [vR] [vL] d vR = vL d d vR
		B0re1 = np.tensordot(np.tensordot(M[i][0], M[i+1][1], axes=(2,0)), ss[l0:,r0:], axes=(3,0))
		B1er0 = np.tensordot(np.tensordot(M[i][2], M[i+1][0], axes=(2,0)), ss[:l0,:r0], axes=(3,0))
		B1ee1 = np.tensordot(np.tensordot(M[i][2], M[i+1][1], axes=(2,0)), ss[l0:,r0:], axes=(3,0))
		B0ee0 = np.tensordot(np.tensordot(M[i][1], M[i+1][2], axes=(2,0)), ss[:l0,:r0], axes=(3,0))
	
		return [B0rr0, B0re1, B1er0, B1ee1, B0ee0]
	def guess_vector_LS_sA(self,i,ss,M):
		l0 = M[i+1][0].shape[2]
		r0 = M[i+2][0].shape[0]
		B0rr10 = np.tensordot(np.tensordot(M[i][0], M[i+1][0], axes=(2,0)), ss[:l0,:r0], axes=(3,0))    # vL d [vR] [vL] d vR = vL d d vR
		B0rr30 = np.tensordot(np.tensordot(M[i][0], M[i+1][1], axes=(2,0)), ss[:l0,:r0], axes=(3,0))
		B0rr21 = np.tensordot(np.tensordot(M[i][0], M[i+1][2], axes=(2,0)), ss[l0:,r0:], axes=(3,0))
		B1er10 = np.tensordot(np.tensordot(M[i][2], M[i+1][0], axes=(2,0)), ss[:l0,:r0], axes=(3,0))
		B1er30 = np.tensordot(np.tensordot(M[i][2], M[i+1][1], axes=(2,0)), ss[:l0,:r0], axes=(3,0))
		B1er21 = np.tensordot(np.tensordot(M[i][2], M[i+1][2], axes=(2,0)), ss[l0:,r0:], axes=(3,0))
		B0er40 = np.tensordot(np.tensordot(M[i][1], M[i+1][3], axes=(2,0)), ss[:l0,:r0], axes=(3,0))
	
		return [B0rr10,B0rr30,B0rr21,B1er10,B1er30,B1er21,B0er40]
	def guess_vector_LS_As(self,i,ss,M):
		l0 = M[i+1][0].shape[2]
		r0 = M[i+2][0].shape[0]
		B0r1r0 = np.tensordot(np.tensordot(M[i][0], M[i+1][0], axes=(2,0)), ss[:l0,:r0], axes=(3,0))    # vL d [vR] [vL] d vR = vL d d vR
		B0r1e1 = np.tensordot(np.tensordot(M[i][0], M[i+1][1], axes=(2,0)), ss[l0:,r0:], axes=(3,0))
		B0r3r0 = np.tensordot(np.tensordot(M[i][1], M[i+1][0], axes=(2,0)), ss[:l0,:r0], axes=(3,0))
		B0r3e1 = np.tensordot(np.tensordot(M[i][1], M[i+1][1], axes=(2,0)), ss[l0:,r0:], axes=(3,0))
		B1r4r0 = np.tensordot(np.tensordot(M[i][3], M[i+1][0], axes=(2,0)), ss[:l0,:r0], axes=(3,0))
		B1r4e1 = np.tensordot(np.tensordot(M[i][3], M[i+1][1], axes=(2,0)), ss[l0:,r0:], axes=(3,0))
		B0r2e0 = np.tensordot(np.tensordot(M[i][2], M[i+1][2], axes=(2,0)), ss[:l0,:r0], axes=(3,0))

		return [B0r1r0,B0r1e1,B0r3r0,B0r3e1,B1r4r0,B1r4e1,B0r2e0]
	def guess_vector_LS_sB(self,i,ss,M):
		l0 = M[i+1][0].shape[2]
		r0 = M[i+2][0].shape[0]
		B0rh20 = np.tensordot(np.tensordot(M[i][0],M[i+1][0], axes=(2,0)), ss[:l0,:r0], axes=(3,0))    # vL d [vR] [vL] d vR = vL d d vR
		B0rh30 = np.tensordot(np.tensordot(M[i][0],M[i+1][1], axes=(2,0)), ss[:l0,:r0], axes=(3,0))
		B0rh40 = np.tensordot(np.tensordot(M[i][0],M[i+1][2], axes=(2,0)), ss[:l0,:r0], axes=(3,0))
		B0rh51 = np.tensordot(np.tensordot(M[i][0],M[i+1][3], axes=(2,0)), ss[l0:,r0:], axes=(3,0))
		B1eh20 = np.tensordot(np.tensordot(M[i][2],M[i+1][0], axes=(2,0)), ss[:l0,:r0], axes=(3,0))
		B1eh30 = np.tensordot(np.tensordot(M[i][2],M[i+1][1], axes=(2,0)), ss[:l0,:r0], axes=(3,0))
		B1eh40 = np.tensordot(np.tensordot(M[i][2],M[i+1][2], axes=(2,0)), ss[:l0,:r0], axes=(3,0))
		B1eh51 = np.tensordot(np.tensordot(M[i][2],M[i+1][3], axes=(2,0)), ss[l0:,r0:], axes=(3,0))
		B0eh10 = np.tensordot(np.tensordot(M[i][1],M[i+1][4], axes=(2,0)), ss[:l0,:r0], axes=(3,0))
	
		return [B0rh20,B0rh30,B0rh40,B0rh51,B1eh20,B1eh30,B1eh40,B1eh51,B0eh10]
	def guess_vector_LS_Bs(self,i,ss,M):
		l0 = M[i+1][0].shape[2]
		r0 = M[i+2][0].shape[0]
		B0h2r0 = np.tensordot(np.tensordot(M[i][0],M[i+1][0], axes=(2,0)), ss[:l0,:r0], axes=(3,0))    # vL d [vR] [vL] d vR = vL d d vR
		B0h2e1 = np.tensordot(np.tensordot(M[i][0],M[i+1][1], axes=(2,0)), ss[l0:,r0:], axes=(3,0))
		B0h3r0 = np.tensordot(np.tensordot(M[i][1],M[i+1][0], axes=(2,0)), ss[:l0,:r0], axes=(3,0))
		B0h3e1 = np.tensordot(np.tensordot(M[i][1],M[i+1][1], axes=(2,0)), ss[l0:,r0:], axes=(3,0))
		B0h4r0 = np.tensordot(np.tensordot(M[i][2],M[i+1][0], axes=(2,0)), ss[:l0,:r0], axes=(3,0))
		B0h4e1 = np.tensordot(np.tensordot(M[i][2],M[i+1][1], axes=(2,0)), ss[l0:,r0:], axes=(3,0))
		B1h1r0 = np.tensordot(np.tensordot(M[i][4],M[i+1][0], axes=(2,0)), ss[:l0,:r0], axes=(3,0))
		B1h1e1 = np.tensordot(np.tensordot(M[i][4],M[i+1][1], axes=(2,0)), ss[l0:,r0:], axes=(3,0))
		B0h5e0 = np.tensordot(np.tensordot(M[i][3],M[i+1][2], axes=(2,0)), ss[:l0,:r0], axes=(3,0))
	
		return [B0h2r0,B0h2e1,B0h3r0,B0h3e1,B0h4r0,B0h4e1,B1h1r0,B1h1e1,B0h5e0]

	# Lanczos algorithm for diagonalisation
	def Lanczos(self,L, R, gv, wi, wj, td, id):
		s=0
		v=[]
		for i in range(len(gv)):
			s = s + gv[i].shape[0] * gv[i].shape[3]
			v.extend(list(gv[i][:,0,0,:].flatten()))
		V = np.zeros((s,1))
		T = np.zeros((1,1))
		v = v/np.linalg.norm(v)
		V[:,0] = v 

		if(id=='ss'):
			w = LHv.Hv(L, R, gv, wi, wj)
		elif(id=='sA'):
			w = LHv.Hv_sA(L, R, gv, wi, wj)
		elif(id=='sB'):
			w = LHv.Hv_sB(L, R, gv, wi, wj)
		elif(id=='As'):
			w = LHv.Hv_As(L, R, gv, wi, wj)
		elif(id=='Bs'):
			w = LHv.Hv_Bs(L, R, gv, wi, wj)

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
				v_j = np.random.rand(s)
				for ii in range(j):
					v_j -= np.dot(v_j, V[:, ii]) * V[:, ii]
				v2 = v_j/np.linalg.norm(v_j)
				print(colored('danger','red'))
			else:
				v2 = w/beta
			t1 = np.zeros((gv[0].shape[0],1,1,gv[0].shape[3]))
			sp = gv[0].shape[0] * gv[0].shape[3]
			t1[:,0,0,:] = v2[:sp].reshape((gv[0].shape[0], gv[0].shape[3]))    
			GV = [t1]
			for q in range(1,len(gv)):
				t = np.zeros((gv[q].shape[0],1,1,gv[q].shape[3]))
				t[:,0,0,:] = v2[sp : sp + gv[q].shape[0]*gv[q].shape[3]].reshape((gv[q].shape[0], gv[q].shape[3]))         
				sp = sp + gv[q].shape[0]*gv[q].shape[3]
				GV.append(t)

			if(id=='ss'):
				w = LHv.Hv(L, R, GV, wi, wj)
			elif(id=='sA'):
				w = LHv.Hv_sA(L, R,GV, wi, wj)
			elif(id=='sB'):
				w = LHv.Hv_sB(L, R, GV, wi, wj)
			elif(id=='As'):
				w = LHv.Hv_As(L, R, GV, wi, wj)
			elif(id=='Bs'):
				w = LHv.Hv_Bs(L, R, GV, wi, wj)
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

		#print('LanczosDiff%i'%j, diff)
		EV = V @ eigV[:,0]                    #This need to be normalised for sum(|si|**2) = 1
		EV = EV/np.linalg.norm(EV)
		E = mev 

		if(id=='ss'):
			return(LHv.FS_ss(gv,EV,E,td))
		elif(id=='sA'):
			return(LHv.FS_sA(gv,EV,E,td))
		elif(id=='sB'):
			return(LHv.FS_sB(gv,EV,E,td))
		elif(id=='As'):
			return(LHv.FS_As(gv,EV,E,td))
		elif(id=='Bs'):
			return(LHv.FS_Bs(gv,EV,E,td))

	# Builidng ladder through inflation rules
	def Openladder_infl(self,n,lm):
		for i in range(n):
			if(i == 0):
				lad = 'sAsAsAsAsAsAsAsA'   #*lm
			elif('B' in lad):
				lad = lad.replace('sA','sAssB!s')
				lad = lad.replace('ssBs','sAsAsAsAssBs')
				lad = lad.replace('B!','B')
			else:
				lad = lad.replace('sA','sAssBs')
		return lad[lm:len(lad)-lm]

	def sweeps(self,n):
		c=0
		Eng = []
		iters = []
		lad = self.Openladder_infl(self.ladN,self.lm)
		N = len(lad)
		print(N)   #lad
		M,d0_r,d1_r = self.exact_states(lad)
		W = self.MPO(lad)
		LE,RE = self.Envs(lad,M,W)
		SS = [ 0,0,0,0,0, np.diag([1/np.sqrt(d0_r+d1_r)]*(d0_r+d1_r))]
		SS.extend([ 0 ]*(N-7))
		td = 100   # Bond Dimension
		for i in range(n):   # Left Sweep
			# print(i)
			for j in range(6,N-6):
				# print(j)
				c=c+1
				id = lad[j:j+2]
				#print(i,j,id) 
				if(id=='ss'):
					gv = self.guess_vector_RS(j,SS[j-1],M)
				elif(id=='sA'):
					gv = self.guess_vector_RS_sA(j,SS[j-1],M)
				elif(id=='sB'):
					gv = self.guess_vector_RS_sB(j,SS[j-1],M)
				elif(id=='As'):
					gv = self.guess_vector_RS_As(j,SS[j-1],M)
				elif(id=='Bs'):
					gv = self.guess_vector_RS_Bs(j,SS[j-1],M)
				E, M[j], M[j+1], SS[j] = self.Lanczos(LE[j-1], RE[j+2], gv, W[j], W[j+1],td,id)

				if(id=='ss'):
					LE[j] = self.LEnv_s(j,M,LE,W)
					RE[j+1] = self.REnv_s(j+1,M,RE,W)
				elif(id=='sA'):
					LE[j] = self.LEnv_s(j,M,LE,W)
					RE[j+1] = self.REnv_A(j+1,M,RE,W)
				elif(id=='sB'):
					LE[j] = self.LEnv_s(j,M,LE,W)
					RE[j+1] = self.REnv_B(j+1,M,RE,W)
				elif(id=='As'):
					LE[j] = self.LEnv_A(j,M,LE,W)
					RE[j+1] = self.REnv_s(j+1,M,RE,W)
				elif(id=='Bs'):
					LE[j] = self.LEnv_B(j,M,LE,W)
					RE[j+1] = self.REnv_s(j+1,M,RE,W)
				Eng.append(E)
				iters.append(c)
				#print("Sweep #%i, RS, site%i"%(i,j), E, '\n')
			td=td+100
			for j in range(N-8,4,-1):   # Right Sweep
				# print(j)
				c=c+1
				id = lad[j:j+2]
				#print(i,j,id) 
				if(id=='ss'):
					gv = self.guess_vector_LS(j,SS[j+1],M)
				elif(id=='sA'):
					gv = self.guess_vector_LS_sA(j,SS[j+1],M)
				elif(id=='sB'):
					gv = self.guess_vector_LS_sB(j,SS[j+1],M)
				elif(id=='As'):
					gv = self.guess_vector_LS_As(j,SS[j+1],M)
				elif(id=='Bs'):
					gv = self.guess_vector_LS_Bs(j,SS[j+1],M)
				E, M[j], M[j+1], SS[j] = self.Lanczos(LE[j-1], RE[j+2], gv, W[j], W[j+1],td,id)
		
				if(id=='ss'):
					LE[j] = self.LEnv_s(j,M,LE,W)
					RE[j+1] = self.REnv_s(j+1,M,RE,W)
				elif(id=='sA'):
					LE[j] = self.LEnv_s(j,M,LE,W)
					RE[j+1] = self.REnv_A(j+1,M,RE,W)
				elif(id=='sB'):
					LE[j] = self.LEnv_s(j,M,LE,W)
					RE[j+1] = self.REnv_B(j+1,M,RE,W)
				elif(id=='As'):
					LE[j] = self.LEnv_A(j,M,LE,W)
					RE[j+1] = self.REnv_s(j+1,M,RE,W)
				elif(id=='Bs'):
					LE[j] = self.LEnv_B(j,M,LE,W)
					RE[j+1] = self.REnv_s(j+1,M,RE,W)
				Eng.append(E)
				iters.append(c)
				#print("Sweep #%i, LS, site%i"%(i,j), E, '\n')
			td=td+100

		return Eng, iters, SS, M, lad

# st = time.time()
# obj = DMRG_AB(J=1,Vr=-7,Vl=10,Vlr=0.01,ladN=8,lm=4)
# Eng, iters, SS, M, lad = obj.sweeps(1)
# print(time.time()-st)
# print(Eng[-1])
# #DD.dimDen(lad,M,SS)
# plt.plot(iters[:],Eng[:])
# plt.xlabel('Iterations')
# plt.ylabel('Energy')
# plt.show()

#vr = [10]*20    #np.linspace(-15,20,25)
#vl = list(np.linspace(-0.6405289533024211,-0.6405129790666025,20))
#ist = list(zip(vr,vl))
#points = []   #[(-15,-20),(-5,-10),(5,-12),(3,-2),(15,-1),(0.1,-20),(10,-1),(15,-20),(25,-2),(25,-10)]
#[(-15,20),(0.1,20),(-10,-4),(-15,-10),(-5,10),(5,11),(15,20),(25,30),(25,35),(25,40)]
#[(15,1),(7.5,1.5),(11,5),(13,4),(15,5),(15,9),(12,1),(25,2),(25,12),(25,20)]
#points.extend(ist)

# for i in vr:
# 	vl = list(np.linspace(i-3, i+3, 5))
# 	#vl = list(np.linspace(i+1, i + 10, 7))
# 	l = list(it.product([i],vl))
# 	points.extend(l)
#
# x,y = zip(*points)
# plt.scatter(*zip(*points),s=20)
# plt.plot(*zip(*ist),c='black')
# plt.axhline(y=0, color='black')
# plt.show()

# st = time.time()
# color = {0:'#7EC0EE',1:'#EEE8AA',2:'#FF82AB',3:'#9A32CD',4:'#7CCD7C'}
for x,y in [(15,20)]:    #points: (vr,vl)   
	print(x,y)
	obj = DMRG_AB(J=1,Vr=x,Vl=y,Vlr=0.01,ladN=5,lm=0)    #Ladder Order = ladN
	Eng, iters, SS, M, lad = obj.sweeps(1)
	# print(time.time()-st)
	plt.plot(iters[:], Eng[:])
	plt.show()
	_,_ = DD.dimDen(lad,M,SS,x,y,0.01)
	DD.EET(lad,SS,x,y,0.01)

	'''
	if(len(Sr)+len(Ar)+len(Br) == N-10 and len(Sl)+len(Al)+len(Bl) == 0):
		print(colored('rungP','blue'))
		plt.plot(x, y, marker="o", markersize=5, markerfacecolor=color[0], markeredgecolor=color[0])
	elif(len(Sl)+len(Al)+len(Bl) == N-10 and len(Sr)+len(Ar)+len(Br) == 0):
		print(colored('colP','blue'))
		plt.plot(x, y, marker="o", markersize=5, markerfacecolor=color[1], markeredgecolor=color[1])
	elif(len(Sl)+len(Ar)+len(Br) == N-10 and len(Sr)+len(Al)+len(Bl) == 0):
		print(colored('rungP3','blue'))
		plt.plot(x, y, marker="o", markersize=5, markerfacecolor=color[2], markeredgecolor=color[2])
	elif(len(Sr)!=0 and len(Sl)!=0 and len(Sr)+len(Sl)+len(Al)+len(Bl) == N-10 and len(Ar)+len(Br) == 0):
		print(colored('rungPs','blue'))
		plt.plot(x, y, marker="o", markersize=5, markerfacecolor=color[3], markeredgecolor=color[3])
	else:
		print(colored(len(Sr)+len(Sl)+len(Al)+len(Ar)+len(Br)+len(Bl), 'blue'))
		print(colored(N-10,'blue'))
	print(colored('Disorder','blue'))
	plt.plot(x, y, marker="o", markersize=5, markerfacecolor=color[4], markeredgecolor=color[4])
	'''
