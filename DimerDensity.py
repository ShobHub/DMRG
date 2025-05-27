import numpy as np
from scipy.stats import linregress
import matplotlib.pylab as plt
import pickle

class dimerDen():
	def dimDen(lad,M,SS,vr,vl,vlr):
		'''
  		M : List of MPS tensors
    		SS : List of Schmidt matrices across the MPS chain
      
		Returns a list of onsite probabilities for finding rung, leg, or rung-leg states on each site of the ladder.

		This helps in understanding the structure of the ground state superposition across different parameter regimes.
		'''
		N = len(lad)
		Prob_r = []
		Prob_l = []
		if(lad[5]=='s'):  # For site 5
			S0 = np.array([[1., 0.], [0., 0.]])   # Onsite Probability Operator for having a rung dimer
			m = M[5]
			l0 = m[0].shape[2]
			s=0
			for j in range(3):  # Finding Onsite Basis Probabilities
				if(j==0):
					x = np.tensordot(np.tensordot(m[j], m[j], axes=(0,0)), S0[:1,:1], axes=([0,2],[1,0]))     #[vL*] d* vR*  [vL] d vR = [d*] vR* [d] vR [Od] [Ou] = vR* vR
				else:
					x = np.tensordot(np.tensordot(m[j], m[j], axes=(0,0)), S0[1:,1:], axes=([0,2],[1,0]))
				if(j!=1):
					ss = np.tensordot(SS[5][:l0,:l0], SS[5][:l0,:l0], axes=(1,1))   # sil* [sir*]  sil [sir] = sil* sil
				else:
					ss = np.tensordot(SS[5][l0:,l0:], SS[5][l0:,l0:], axes=(1,1))
				f = np.tensordot(x, ss, axes=([0,1],[0,1]))           # [vR*] [vR]  [sil*] [sil]
				s = s + f
			#print("Site is %s at %i"%('s',5), s)
			Prob_r.append(s)    # Probabilities of having rung states on 's'
			Prob_l.append(1-s)  # Probabilities of having leg states on 's'
		elif(lad[5] == 'A'):
			m = M[5]
			l0 = m[0].shape[2]
			OP = []
			for op in range(4):   # Finding 4 Onsite Basis Probabilities
				a0 = np.array([[0]])
				a1 = np.array([[1]])
				s=0
				for j in range(4):
					if(op==0 and j==0):
						x = np.tensordot(np.tensordot(m[j], m[j], axes=(0,0)), a1, axes=([0,2],[1,0]))
					elif(op==1 and j==2):
						x = np.tensordot(np.tensordot(m[j], m[j], axes=(0,0)), a1, axes=([0,2],[1,0]))
					elif(op==2 and j==1):
						x = np.tensordot(np.tensordot(m[j], m[j], axes=(0,0)), a1, axes=([0,2],[1,0]))
					elif(op==3 and j==3):
						x = np.tensordot(np.tensordot(m[j], m[j], axes=(0,0)), a1, axes=([0,2],[1,0]))
					else:
						x = np.tensordot(np.tensordot(m[j], m[j], axes=(0,0)), a0, axes=([0,2],[1,0]))
					if(j!=2):
						ss = np.tensordot(SS[5][:l0,:l0], SS[5][:l0,:l0], axes=(1,1))
					else:
						ss = np.tensordot(SS[5][l0:,l0:], SS[5][l0:,l0:], axes=(1,1))
					f = np.tensordot(x, ss, axes=([0,1],[0,1]))
					s += f
				OP.append(s)
				#print("Site is %s at %i and OP is %i"%('A',5,op), s)
			Prob_r.append(OP[0]+OP[2])   # Probabilities of having leg-rung states on 'A'
			Prob_l.append(OP[1]+OP[3])   # Probabilities of having all leg states on 'A'
		elif(lad[5] == 'B'):
			m = M[5]
			l0 = m[0].shape[2]
			OP = []
			for op in range(5):   # Finding 5 Onsite Basis Probabilities
				b0 = np.array([[0]])
				b1 = np.array([[1]])
				s=0
				for j in range(5):
					if(op==0 and j==4):
						x = np.tensordot(np.tensordot(m[j], m[j], axes=(0,0)), b1, axes=([0,2],[1,0]))
					elif(op==1 and j==0):
						x = np.tensordot(np.tensordot(m[j], m[j], axes=(0,0)), b1, axes=([0,2],[1,0]))
					elif(op==2 and j==1):
						x = np.tensordot(np.tensordot(m[j], m[j], axes=(0,0)), b1, axes=([0,2],[1,0]))
					elif(op==3 and j==2):
						x = np.tensordot(np.tensordot(m[j], m[j], axes=(0,0)), b1, axes=([0,2],[1,0]))
					elif(op==4 and j==3):
						x = np.tensordot(np.tensordot(m[j], m[j], axes=(0,0)), b1, axes=([0,2],[1,0]))
					else:
						x = np.tensordot(np.tensordot(m[j], m[j], axes=(0,0)), b0, axes=([0,2],[1,0]))
					if(j!=3):
						ss = np.tensordot(SS[5][:l0,:l0], SS[5][:l0,:l0], axes=(1,1))   # [sil*] sir*  [sil] sir = sir* sir
					else:
						ss = np.tensordot(SS[5][l0:,l0:], SS[5][l0:,l0:], axes=(1,1))
					f = np.tensordot(x, ss, axes=([0,1],[0,1]))
					s += f
				OP.append(s)
				#print("Site is %s at %i and OP is %i"%('B',5,op), s)
			Prob_r.append(OP[1]+OP[2]+OP[3]) # Probabilities of having leg-rung states on 'B'
			Prob_l.append(OP[0]+OP[4])       # Probabilities of having all leg states on 'B'
			
		for i in range(6,N-5):   # For rest of ladder
			if(lad[i]=='s'):
				S0 = np.array([[1., 0.], [0., 0.]])
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
				#print("Site is %s at %i"%('s',i), s)
				Prob_r.append(s)
				Prob_l.append(1-s)
			elif(lad[i] == 'A'):
				m = M[i]
				r0 = m[0].shape[0]
				OP = []
				for op in range(4):
					a0 = np.array([[0]])
					a1 = np.array([[1]])
					s=0
					for j in range(4):
						if(op==0 and j==0):
							x = np.tensordot(np.tensordot(m[j], m[j], axes=(2,2)), a1, axes=([1,3],[1,0]))     
						elif(op==1 and j==2): 
							x = np.tensordot(np.tensordot(m[j], m[j], axes=(2,2)), a1, axes=([1,3],[1,0])) 
						elif(op==2 and j==1): 
							x = np.tensordot(np.tensordot(m[j], m[j], axes=(2,2)), a1, axes=([1,3],[1,0]))
						elif(op==3 and j==3): 
							x = np.tensordot(np.tensordot(m[j], m[j], axes=(2,2)), a1, axes=([1,3],[1,0])) 
						else:
							x = np.tensordot(np.tensordot(m[j], m[j], axes=(2,2)), a0, axes=([1,3],[1,0]))
						if(j!=3):
							ss = np.tensordot(SS[i-1][:r0,:r0], SS[i-1][:r0,:r0], axes=(0,0))   # [sil*] sir*  [sil] sir = sir* sir
						else:
							ss = np.tensordot(SS[i-1][r0:,r0:], SS[i-1][r0:,r0:], axes=(0,0)) 
						f = np.tensordot(ss, x, axes=([0,1],[0,1]))  
						s += f
					OP.append(s)
					#print("Site is %s at %i and OP is %i"%('A',i,op), s)
				Prob_r.append(OP[0]+OP[2])
				Prob_l.append(OP[1]+OP[3])
			elif(lad[i] == 'B'):
				m = M[i]
				r0 = m[0].shape[0]
				OP = []
				for op in range(5):
					b0 = np.array([[0]])
					b1 = np.array([[1]])
					s=0
					for j in range(5):
						if(op==0 and j==4):
							x = np.tensordot(np.tensordot(m[j], m[j], axes=(2,2)), b1, axes=([1,3],[1,0]))     
						elif(op==1 and j==0): 
							x = np.tensordot(np.tensordot(m[j], m[j], axes=(2,2)), b1, axes=([1,3],[1,0])) 
						elif(op==2 and j==1): 
							x = np.tensordot(np.tensordot(m[j], m[j], axes=(2,2)), b1, axes=([1,3],[1,0]))
						elif(op==3 and j==2): 
							x = np.tensordot(np.tensordot(m[j], m[j], axes=(2,2)), b1, axes=([1,3],[1,0])) 
						elif(op==4 and j==3): 
							x = np.tensordot(np.tensordot(m[j], m[j], axes=(2,2)), b1, axes=([1,3],[1,0])) 
						else:
							x = np.tensordot(np.tensordot(m[j], m[j], axes=(2,2)), b0, axes=([1,3],[1,0]))
						if(j!=4):
							ss = np.tensordot(SS[i-1][:r0,:r0], SS[i-1][:r0,:r0], axes=(0,0))   # [sil*] sir*  [sil] sir = sir* sir
						else:
							ss = np.tensordot(SS[i-1][r0:,r0:], SS[i-1][r0:,r0:], axes=(0,0)) 
						f = np.tensordot(ss, x, axes=([0,1],[0,1]))  
						s += f
					OP.append(s)
					#print("Site is %s at %i and OP is %i"%('B',i,op), s)
				Prob_r.append(OP[1]+OP[2]+OP[3])
				Prob_l.append(OP[0]+OP[4])
		# plt.clf()
		# plt.plot(np.arange(5,N-6,1), Prob_r[:-1], label='rung')
		# plt.scatter(np.arange(5,N-6,1), Prob_r[:-1], s=8)
		# plt.plot(np.arange(5,N-6,1), Prob_l[:-1], label='leg')
		# plt.scatter(np.arange(5,N-6,1), Prob_l[:-1], s=8)
		# plt.title('Vr=%0.02f;Vl=%0.02f;Vlr=%0.02f' % (vr, vl, vlr))
		# plt.ylim(-0.1,1.1)
		# plt.xlabel('Ladder sites')
		# plt.ylabel('Probability')
		# plt.legend()
		# plt.show()
		# pickle.dump([Prob_r,Prob_l],open('Prob/Lad5/Vr=%0.02f_Vl=%0.02f_Vlr=%0.02f.txt'%(vr,vl,vlr),'wb'))
		# # plt.savefig('Prob/Vlr=-30_Lad2/Vr=%0.02f_Vl=%0.02f_Vlr=%0.02f.pdf'%(vr,vl,vlr))
		# print(vr,vl,lad)
		# print(Prob_r)
		# print(Prob_l)
		return Prob_r,Prob_l

	def EET(lad,SS,vr,vl,vlr):   # Entanglement Entropy 
		'''
  		SS : List of Schmidt matrices across the MPS chain
  		Returns the list of Entanglement Entropies at each bond across MPS
    		'''
		ET = []
		N = len(lad)
		cd = {}
		for i in range(5,N-6):
			s = 0
			for j in SS[i].diagonal():
				s = s + (j**2) * np.log(j**2)
			ET.append(-s)
			cd[(2*N/np.pi)*np.sin(np.pi*i/N)] = -s
		plt.clf()
		plt.plot(np.arange(5,N-6,1),ET)
		plt.scatter(np.arange(5,N-6,1),ET,s=8)
		plt.title('Vr=%0.02f;Vl=%0.02f;Vlr=%0.02f'%(vr,vl,vlr))
		plt.xlabel('Ladder sites')
		plt.ylabel('Entanglement Entropy')
		pickle.dump(ET, open('EntanglementEntropy/Lad5/Vr=%0.02f_Vl=%.9f_Vlr=%0.02f.txt' % (vr, vl, vlr), 'wb'))
		# plt.savefig('EntanglementEntropy/Lad3/CDvsET/Vr=%0.02f_Vl=%.9f_Vlr=%0.02f.pdf' % (vr, vl, vlr))  # Vlr=-30_Lad2
		plt.show()

		#pickle.dump(cd,open('EntanglementEntropy/Lad3/CDvsET/Sn_Vr=%0.02f_Vl=%.9f_Vlr=%0.02f.txt'%(vr,vl,vlr), 'wb'))
		# cd_n = dict(sorted(cd.items()))
		# cdk = list(cd_n.keys())
		# cdv = list(cd_n.values())
		# sl,intc,_,_, std_err = linregress(np.log(cdk[:]),cdv[:])
		# print(sl,std_err)
		# plt.scatter(cdk, cdv, s=8)
		# plt.plot(cdk[:], sl*np.log(cdk[:])+intc)
		# plt.title('Vr=%0.02f;Vl=%0.02f;Vlr=%0.02f' % (vr, vl, vlr))
		# plt.xlabel('Conformal Distance d(n)')
		# plt.ylabel('Entanglement Entropy')
		# plt.xscale('log')
		#plt.xlim(10,400)
		#plt.show()

	def CorreFn(lad,M,SS,p,q):   
		N = len(lad)
		li0 = M[5][0].shape[0]
		li1 = M[5][2].shape[0]
		prev = [np.eye(li0),np.eye(li1)]
		S1 = np.array([[0., 0.], [0., 1.]])
		SL_A = np.array([[0., 0., 0., 0.],[0., 0., 0., 0.],[0., 0., 1., 0.],[0., 0., 0., 1.]])
		SL_B= np.array([[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [0., 0., 0., 1., 0.],[0., 0., 0., 0., 1.]])
		for i in range(5,N-6):
			if (lad[i]=='s'):
				l0 = M[i][0].shape[2]
				b0 = [[['0 0'],['2 2']],[['1 1'],['*']]]
				if i==p or i==q:
					O = S1
				else:
					O = np.eye(2)
				current = []
				for m in range(2):
					s = 0
					for j in range(2):
						for k in range(len(b0[m][j])):
							if (b0[m][j][k] != '*'):
								a,b = b0[m][j][k].split()
								x = np.tensordot(np.tensordot(prev[j], M[i][int(b)], axes=(0,0)),M[i][int(a)],axes=(0,0))       #[D] U  [vl] d vR = [U] d vR  [vl*] d* vR* = d vR d* vR*
								if a!='0':
									x = np.tensordot(x,O[1:,1:],axes=([0, 2], [0, 1]))   #[d] vR [d*] vR*  [Od] [Ou] = vR vR*
								else:
									x = np.tensordot(x,O[:1,:1],axes=([0, 2], [0, 1]))
								if a!='1':
									x = np.tensordot(np.tensordot(x,SS[i][:l0,:l0],axes=(0,0)),SS[i][:l0,:l0],axes=(0,0))    #[vR] vR*  [siL] siR = [vR*] siR  [siL*] siR* = siR siR*
								else:
									x = np.tensordot(np.tensordot(x,SS[i][l0:,l0:],axes=(0,0)),SS[i][l0:,l0:],axes=(0,0))
								s = s+x
					current.append(s)
				prev = current

			if (lad[i] == 'A'):
				l0 = M[i][0].shape[2]
				b0 = [[['0 0','1 1'], ['3 3']], [['2 2'], ['*']]]
				if i == p or i == q:
					O = SL_A
				else:
					O = np.eye(4)
				current = []
				for m in range(2):
					s = 0
					for j in range(2):
						for k in range(len(b0[m][j])):
							if (b0[m][j][k] != '*'):
								a, b = b0[m][j][k].split()
								x = np.tensordot(np.tensordot(prev[j], M[i][int(b)], axes=(0, 0)), M[i][int(a)],axes=(0, 0))  # [D] U  [vl] d vR = [U] d vR  [vl*] d* vR* = d vR d* vR*
								x = np.tensordot(x, O[int(a):int(a)+1, int(a):int(a)+1],axes=([0, 2], [0, 1]))  # [d] vR [d*] vR*  [Od] [Ou] = vR vR*
								if a!='2':
									x = np.tensordot(np.tensordot(x,SS[i][:l0,:l0],axes=(0,0)),SS[i][:l0,:l0],axes=(0,0))    #[vR] vR*  [siL] siR = [vR*] siR  [siL*] siR* = siR siR*
								else:
									x = np.tensordot(np.tensordot(x,SS[i][l0:,l0:],axes=(0,0)),SS[i][l0:,l0:],axes=(0,0))
								s = s + x
					current.append(s)
				prev = current

			if (lad[i] == 'B'):
				l0 = M[i][0].shape[2]
				b0 = [[['0 0', '1 1', '2 2'], ['4 4']], [['3 3'], ['*']]]
				if i == p or i == q:
					O = SL_B
				else:
					O = np.eye(5)
				current = []
				for m in range(2):
					s = 0
					for j in range(2):
						for k in range(len(b0[m][j])):
							if (b0[m][j][k] != '*'):
								a, b = b0[m][j][k].split()
								x = np.tensordot(np.tensordot(prev[j], M[i][int(b)], axes=(0, 0)), M[i][int(a)], axes=(0, 0))  # [D] U  [vl] d vR = [U] d vR  [vl*] d* vR* = d vR d* vR*
								x = np.tensordot(x, O[int(a):int(a)+1, int(a):int(a)+1], axes=([0, 2], [0, 1]))  # [d] vR [d*] vR*  [Od] [Ou] = vR vR*
								if a!='3':
									x = np.tensordot(np.tensordot(x,SS[i][:l0,:l0],axes=(0,0)),SS[i][:l0,:l0],axes=(0,0))    #[vR] vR*  [siL] siR = [vR*] siR  [siL*] siR* = siR siR*
								else:
									x = np.tensordot(np.tensordot(x,SS[i][l0:,l0:],axes=(0,0)),SS[i][l0:,l0:],axes=(0,0))
								s = s + x
					current.append(s)
				prev = current
		result = np.tensordot(prev[0],np.eye(prev[0].shape[0]), axes=([0, 1], [0, 1])) + np.tensordot(prev[1],np.eye(prev[1].shape[0]), axes=([0, 1], [0, 1]))
		return result


## PLOT ONSITE PROBILITIES PLOT & EE
Lad = pickle.load(open('AllData/YtoG/Lad_L=432.txt','rb'))[5:]
N = len(Lad)
i = 80   #201    #80
j = 110  #231    #110
print(Lad[i:j])

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(170,170)) #, sharex=True, sharey=False)  ncols=5, (750,130)

str = ['I','II','III','IV','V']
cap = ['(b)','(c)','(d)','(e)','(f)']
Vs = [(-15,20),(15,-20),(15,12),(-15,-10),(15,20)]
for idx, ax in [(4,axs)]:  #enumerate(axs):
	Vr, Vl = Vs[idx]
	Vlr = 0.01
	Prob = pickle.load(open('Prob/Lad4/Vr=%.2f_Vl=%.2f_Vlr=%.2f.txt' % (Vr, Vl, Vlr), 'rb'))
	Prob_r = Prob[0]
	Prob_l = Prob[1]
	for n, txt in enumerate(Lad[i:j]):
		ax.annotate(txt, (i + n, 1.1), ha='center', fontsize=540)
	ax.plot(np.arange(i, j, 1), Prob_r[i:j], linewidth=30, linestyle = '--', label=r'{s: $\langle n^s \rangle$, A: $\langle O_1^A + O_3^A \rangle$,}'+'\n'+r'{B: $\langle O_2^B + O_3^B + O_4^B \rangle$}')
	ax.scatter(np.arange(i, j, 1), Prob_r[i:j], s=18000)
	ax.plot(np.arange(i, j, 1), Prob_l[i:j], linewidth=30, linestyle = '--', label=r'{s: $\langle I-n^s \rangle$, A: $\langle O_2^A + O_4^A \rangle$,}'+'\n'+r'{B: $\langle O_1^B + O_5^B \rangle$}')
	ax.scatter(np.arange(i, j, 1), Prob_l[i:j], s=18000)
	# ax.plot([], [], alpha=0, label='(Vr, Vl, Vlr) = (%0.02f, %0.02f, %0.02f)' % (Vr, Vl, Vlr))
	ax.set_title(r'Phase %s, $\Lambda_4$'%str[idx], fontsize = 600, y=1.03)
	# ax.text(0.03, 1.1, cap[idx], transform=ax.transAxes, fontsize=550, fontweight='bold', va='top', ha='left')

	if idx == 0:
		ax.legend(fontsize = 560,loc='center right') #,ncol=2)
	ax.tick_params(axis='x', labelsize=485, length=150, width=30)
	ax.tick_params(axis='y', labelsize=490, length=150, width=30)
	ax.set_ylim(-0.1, 1.2)

fig.text(0.55, 0.0001, r'Ladder sites', fontsize=650, ha='center')
fig.text(0.0094, 0.5, r'Probability', fontsize=650, rotation='vertical', va='center')
plt.subplots_adjust(wspace=0.03)
plt.savefig('Prob/ProbDen_Lad=4_5.pdf')
# plt.show()

# for idx, ax in enumerate(axs):
# 	Vr, Vl = Vs[idx]
# 	Vlr = 0.01
# 	EE = pickle.load(open('EntanglementEntropy/Lad5/Vr=%.2f_Vl=%.9f_Vlr=%.2f.txt' % (Vr, Vl, Vlr), 'rb'))
# 	for n, txt in enumerate(Lad[i:j]):
# 		ax.annotate(txt, (i+n, EE[i+n]), ha='center', fontsize=450)
# 	ax.plot(np.arange(i, j, 1), EE[i:j], linewidth=28, linestyle = '--')
# 	ax.scatter(np.arange(i, j, 1), EE[i:j], s=14000)
# 	ax.set_title(r'Phase %s, LadN = 5'%str[idx], fontsize = 550, y=1.03)
# 	ax.text(0.03, 1.1, cap[idx], transform=ax.transAxes, fontsize=550, fontweight='bold', va='top', ha='left')
# 	ax.tick_params(axis='x', labelsize=460, length=150, width=30)
# 	ax.tick_params(axis='y', labelsize=480, length=150, width=30)
# 	# ax.set_ylim(-0.1, 1.2)

# fig.text(0.55, 0.0001, r'Ladder sites', fontsize=650, ha='center')
# fig.text(0.082, 0.5, r'Entanglement Entropy', fontsize=650, rotation='vertical', va='center')
# plt.subplots_adjust(wspace=0.18)
# plt.savefig('EntanglementEntropy/EE_Lad=5.pdf')

fgfdhg

## PLOT Phase Transitions
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# fig, axs = plt.subplots(figsize=(5,5))
color = ['green','blue']
mk = ['o','s']
# xf = [-0.56,-0.58]
# m = 0
# for i in [(4,432),(5,1296)]:
# 	c = 0
# 	for j in [8,20]:
# 		VL, OP = pickle.load(open('YelToPink/E_N/Vr=%.3f_L=%i.txt' % (j, i[1]), 'rb'))
# 		plt.scatter(VL, OP, s=500, marker = mk[m], facecolors='none', edgecolors=color[c], linewidths=2.5, label = '$v_r$=%.2f, $LadN$ = %i'%(j,i[0]))
# 		plt.axvline(x=xf[c], color=color[c], linestyle='--',linewidth=3)
# 		c = c + 1
# 	m = m + 1
# plt.ylabel(r'OP', fontsize = 40)
# plt.xlabel(r'$v_l$',fontsize = 40)
# plt.tick_params(axis='x', labelsize=40, length=10, width=2)
# plt.tick_params(axis='y', labelsize=40, length=10, width=2)
# plt.legend(fontsize = 30, loc='upper right',frameon=False,handletextpad=0.001)
# plt.savefig('PT_YtoP_EN.pdf')
# # plt.show()
#
# gffd

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(30,15), sharex=False, sharey=False)
xf = [-24.20,-3.75]
vr = [9,20]
for idx, ax in enumerate(axs):
	m = 0
	for i in [(4,432),(5,1296)]:
		VL, OP = pickle.load(open('PtoDB/E_N/Vr=%.3f_L=%i.txt' % (vr[idx], i[1]), 'rb'))
		ax.scatter(VL, OP, s=1500, marker = mk[m], facecolors='none', edgecolors=color[idx], linewidths=4, label = '$LadN$ = %i'%i[0])
		# ax.axvline(x=xf[idx], color=color[idx], linestyle='--',linewidth=2)
		m = m + 1
	ax.set_title(r'$v_r$ = %.2f' % vr[idx], fontsize=120, y=1.03)
	ax.legend(fontsize=80, bbox_to_anchor=(0.5, 0.28),frameon=False,handletextpad=0.001)  # ,ncol=2)  loc='lower left'
	ax.tick_params(axis='x', labelsize=80, length=40, width=8)
	ax.tick_params(axis='y', labelsize=80, length=40, width=8)

fig.text(0.32, 0.015, r'$v_l$', fontsize=110, ha='center')
fig.text(0.72, 0.015, r'$v_l$', fontsize=110, ha='center')
fig.text(0.01, 0.5, r'$E_N/N$', fontsize=100, rotation='vertical', va='center')  #OP
plt.subplots_adjust(wspace=0.2)  #0.05
plt.savefig('PT_PtoDB_EN.pdf')
# plt.show()
