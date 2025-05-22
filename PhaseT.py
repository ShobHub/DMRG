import matplotlib.pyplot as plt
import numpy as np
import itertools as it
from scipy import stats
from DimerDensity import dimerDen as DD
from QDM_Initialisation_AB import DMRG_AB as dm
import pickle

class phases():
    def closeB(self,lad,j):
        ra = lad[j:j+10].index('B')
        la = lad[j-10:j][::-1].index('B')
        if ra<la:
            return j+ra
        elif la<ra:
            return j-la-1

    def OP_YtoP(self,N,vr,VL,str):
        OP = []
        lad = pickle.load(open('AllData/%s/Lad_L=%i.txt'%(str,N),'rb'))
        for j in VL:
            Prob_l = pickle.load(open('AllData/%s/ProbL_Vr=%.3f_Vl=%.5f_L=%i.txt' %(str,vr,j,N),'rb'))
            if lad[int(N/2)] == 'B':
                # print(int(l/2)-5)
                OP.append(Prob_l[int(N/2)-5])
            else:
                j = self.closeB(lad,int(N/2))
                print(j,lad[j])
                OP.append(Prob_l[j-5])
        return OP

    def OP_PtoDB(self,N,vr,VL,str,sub):
        OP = []
        lad = pickle.load(open('AllData/%s/Lad_L=%i.txt' % (str, N), 'rb'))
        ind = []
        for i in range(5,len(lad)-5):
            if lad[i:i+2]=='ss' and lad[i-3:i]!='AsA' and lad[i+2:i+5]!='AsA':
                ind.extend([i-5,i+1-5])
        for j in VL:
            Prob_r = pickle.load(open('AllData/%s/ProbR_Vr=%.3f_Vl=%.5f_L=%i.txt' %(str,vr,j,N),'rb'))
            sum = 0
            for i in ind:
                sum += Prob_r[i]
            OP.append(sum/(len(ind)/2) - sub)
        return OP

    def Correlation_Length(self,N,vr,VL,str):
        lad = pickle.load(open('AllData/%s/Lad_L=%i.txt' % (str,N), 'rb'))
        for j in VL[:]:
            M = pickle.load(open('AllData/%s/FS_Vr=%.3f_Vl=%.5f_L=%i.txt' % (str, vr, j, N), 'rb'))
            SS = pickle.load(open('AllData/%s/SS_Vr=%.3f_Vl=%.5f_L=%i.txt' % (str, vr, j, N), 'rb'))
            Prob_l = pickle.load(open('AllData/%s/ProbL_Vr=%.3f_Vl=%.5f_L=%i.txt' % (str, vr, j, N), 'rb'))
            half = int(N/2)
            Cij = []
            x = []
            for i in range(half+1,N-10):
                print(i)
                Cij.append(DD.CorreFn(lad,M,SS,i,half) - (Prob_l[i] * Prob_l[half]))  ##??
                x.append(i-half)
            plt.scatter(x,Cij,s=10)
            plt.show()

    def energyN_PT(self,l,pf,pc,vlFlag,str,lm):            # if vlFlag = True --> parameter changing(pc) along a PT line is vl and vr is the fixed parameter(pf)
        E = []
        for j in pc:
            print(j)
            if vlFlag:
                obj = dm(J=1,Vr=pf, Vl=j, Vlr=vlr, ladN=l, lm=lm)
                Eng, iters, SS, M, lad = obj.sweeps(1)
                Prob_r, Prob_l = DD.dimDen(lad,M,SS,pf,j,vlr)
                pickle.dump([iters, Eng], open('AllData/%s/EvsIters_Vr=%.3f_Vl=%.5f_L=%i.txt' % (str,pf,j,len(lad)), 'wb'))
                pickle.dump(SS, open('AllData/%s/SS_Vr=%.3f_Vl=%.5f_L=%i.txt' % (str,pf,j,len(lad)), 'wb'))
                pickle.dump(M, open('AllData/%s/FS_Vr=%.3f_Vl=%.5f_L=%i.txt' % (str,pf,j,len(lad)), 'wb'))
                pickle.dump(Prob_r, open('AllData/%s/ProbR_Vr=%.3f_Vl=%.5f_L=%i.txt' % (str,pf,j,len(lad)), 'wb'))
                pickle.dump(Prob_l, open('AllData/%s/ProbL_Vr=%.3f_Vl=%.5f_L=%i.txt' % (str, pf, j, len(lad)), 'wb'))
            else:
                obj = dm(J=1, Vr=j, Vl=pf, Vlr=vlr, ladN=l, lm=lm)
                Eng, iters, SS, M, lad = obj.sweeps(4)
                Prob_r, Prob_l = DD.dimDen(lad,M,SS,j,pf,vlr)
                pickle.dump([iters, Eng], open('AllData/%s/EvsIters_Vr=%.3f_Vl=%.5f_L=%i.txt' % (str,j,pf,len(lad)), 'wb'))
                pickle.dump(SS, open('AllData/%s/SS_Vr=%.3f_Vl=%.5f_L=%i.txt' % (str,j,pf,len(lad)), 'wb'))
                pickle.dump(M, open('AllData/%s/FS_Vr=%.3f_Vl=%.5f_L=%i.txt' % (str,j,pf,len(lad)), 'wb'))
                pickle.dump(Prob_r, open('AllData/%s/ProbR_Vr=%.3f_Vl=%.5f_L=%i.txt' % (str,j,pf,len(lad)), 'wb'))
                pickle.dump(Prob_l, open('AllData/%s/ProbL_Vr=%.3f_Vl=%.5f_L=%i.txt' % (str,j,pf,len(lad)), 'wb'))
            E.append(Eng[-1]/len(lad))
            print(len(lad))
            # plt.plot(iters[:], Eng[:])
            # plt.show()
        pickle.dump(lad, open('AllData/%s/Lad_L=%i.txt' % (str,len(lad)), 'wb'))
        return E,len(lad),lad

    def UniClass(self, VR, VL, str_):
        #LM = [148,115,92,60,33]   #133,
        LM = [445, 345, 275, 180] #, 400, 100]
        OP_uc = np.zeros((len(VL),len(LM)))
        LAD = []
        for lm in range(len(LM)):
            _,_,lad = self.energyN_PT(5,VR,VL,True,str_,LM[lm])
            LAD.append(len(lad))
            ind = []
            for i in range(5, len(lad) - 5):
                if lad[i:i + 2] == 'ss' and lad[i - 3:i] != 'AsA' and lad[i + 2:i + 5] != 'AsA':
                    ind.extend([i - 5, i + 1 - 5])
            mid = int(len(ind)/2)
            for j in range(len(VL)):
                Prob_r = pickle.load(open('AllData/%s/ProbR_Vr=%.3f_Vl=%.5f_L=%i_1.txt' %(str_, VR, VL[j], len(lad)), 'rb'))
                print(len(Prob_r))
                for i in ind[mid-2:mid+2]:   #[2:-2]
                    OP_uc[j][lm] += Prob_r[i]
                OP_uc[j][lm] = OP_uc[j][lm]/2     #(mid-2)  #2
            print(OP_uc,np.log(OP_uc))
        return OP_uc,LAD

vlr = 0.01
ladL = [4]
obj = phases()
Col = {}
for i in range(len(ladL)):
    Col[ladL[i]] = np.random.choice(range(256), size=3)/256

# Yellow to Pink
VR = [9]
VL = np.linspace(-0.625,-0.47,40)      # (2,3,5,10,20)[-0.75,-0.3,40] ; (0.1)[-1.5,-0.04,40] (20)(-0.625,-0.54,35)
for i in VR:
    for j in ladL:
        #obj.Correlation_Length(1296, i, VL, 'YelToPink')
        E,N,_ = obj.energyN_PT(j,i,VL,True,'YelToPink',0)
        OP = obj.OP_YtoP(N,i,VL,'YelToPink')
        print(E)
        print(OP)
        pickle.dump([VL,E],open('YelToPink/E_N/Vr=%.3f_L=%i.txt'%(i,N),'wb'))
        pickle.dump([VL,OP], open('YelToPink/OP/Vr=%.3f_L=%i.txt' % (i,N), 'wb'))
        # VL,E = pickle.load(open('YelToPink/E_N/Vr=%.3f_L=%i.txt' % (i, 1296), 'rb'))
        plt.scatter(VL,OP, s=50, facecolors='none',edgecolors=Col[j],label='%i'%N)
    plt.title('Vr=%.3f'%i)
    plt.legend()
    plt.xlabel('vl')
    plt.ylabel('E_N/N')
    plt.show()
ghghjgjh

# Green to Light Blue
# VR = [-6]
# VL = np.linspace(-5,-2.3,40)    #(-3)[-2.8,-1.3,40] ; (-40)[-25,-23,40] ; (-30)[-19,-16.5,40] ; (-20)[-12.75,-11.5,40] ; (-10)[-7.5,-4.5,40] ; (-5)[-4.25,-2,40]
# for i in VR:
#     for j in ladL:
#         E,N,_ = obj.energyN_PT(j,i,VL,True,'GtoLB',0)   #add lm=0 for full ladder length
#         OP = obj.OP_YtoP(N,i,VL,'GtoLB')
#         # obj.Correlation_Length(1296, i, VL, 'GtoLB')
#         print(E)
#         print(OP)
#         pickle.dump([VL,E],open('GtoLB/E_N/Vr=%.3f_L=%i.txt'%(i,N),'wb'))
#         pickle.dump([VL,OP], open('GtoLB/OP/Vr=%.3f_L=%i.txt' % (i,N), 'wb'))
#         # _,E = pickle.load(open('GtoLB/E_N/Vr=%.3f_L=%i.txt' % (i, 1296), 'rb'))
#         plt.scatter(VL,OP, s=50, facecolors='none', edgecolors=Col[j],label='%i'%N)
#     plt.title('Vr=%.3f'%i)
#     plt.legend()
#     plt.xlabel('vl')
#     plt.ylabel('E_N/N')
#     plt.show()

#YtoGtoLB
#Vr = -5 ; Vl = (-7.2,-1,50)

# Pink to Dark Blue
# VR = [9]
# VL = np.linspace(5.5,14.5,40)     #(20)[17,26,40] (5)[2.5,8.5,40] (15)[12,21.5,40]
# for i in VR:
#     for j in ladL:
#         E,N,_ = obj.energyN_PT(j,i,VL,True,'PtoDB',0)
#         OP = obj.OP_PtoDB(N,i,VL,'PtoDB',0)
#         print(E)
#         print(OP)
#         pickle.dump([VL,E],open('PtoDB/E_N/Vr=%.3f_L=%i.txt'%(i,N),'wb'))
#         pickle.dump([VL,OP], open('PtoDB/OP/Vr=%.3f_L=%i.txt' % (i,N), 'wb'))
#         # VL,OP = pickle.load(open('PtoDB/OP/Vr=%.3f_L=%i.txt' % (i, 432), 'rb'))
#         plt.scatter(VL,OP, s=50, facecolors='none', edgecolors=Col[j],label='%i'%N)
#     plt.title('Vr=%.3f'%i)
#     plt.legend()
#     plt.xlabel('vl')
#     plt.ylabel('E_N/N')
#     plt.show()

# Yellow to Green
# VR = [-8]
# VL = np.linspace(-10.8,-5.3,40)     #(-20)[-25.5,-15.5,40] (-15)[-21,-10,40] (-5)[-9,-3.4,40]
# for i in VR:
#     for j in ladL:
#         E,N,_ = obj.energyN_PT(j,i,VL,True,'YtoG',0)
#         OP = obj.OP_PtoDB(N,i,VL,'YtoG',0)
#         print(E)
#         print(OP)
#         pickle.dump([VL,E],open('YtoG/E_N/Vr=%.3f_L=%i.txt'%(i,N),'wb'))
#         pickle.dump([VL,OP], open('YtoG/OP/Vr=%.3f_L=%i.txt' % (i,N), 'wb'))
#         plt.scatter(VL,OP, s=50, facecolors='none', edgecolors=Col[j],label='%i'%N)
#     plt.title('Vr=%.3f'%i)
#     plt.legend()
#     plt.xlabel('vl')
#     plt.ylabel('E_N/N')
#     plt.show()

# Dark Blue to Light Blue
# VR = [9]
# VL = np.linspace(13.8,23,40)     #(10)[15,25,40] ; (15)[23,36,40] ; (5)[7.8,16,40] ; (18)[31,40,40]
# for i in VR:
#     for j in ladL:
#         E,N,_ = obj.energyN_PT(j,i,VL,True,'DBtoLB',0)
#         OP = obj.OP_PtoDB(N,i,VL,'DBtoLB',1)
#         print(E)
#         print(OP)
#         pickle.dump([VL,E],open('DBtoLB/E_N/Vr=%.3f_L=%i.txt'%(i,N),'wb'))
#         pickle.dump([VL,OP], open('DBtoLB/OP/Vr=%.3f_L=%i.txt' % (i,N), 'wb'))
#         plt.scatter(VL,OP, s=50, facecolors='none', edgecolors=Col[j],label='%i'%N)
#     plt.title('Vr=%.3f'%i)
#     plt.legend()
#     plt.xlabel('vl')
#     plt.ylabel('E_N/N')
#     plt.show()

'''
# P to DB (UNI CLASS)
vlr = 0.01
VL = np.linspace(21.48,21.62,3)
obj = phases()

# OP_uc,LAD = obj.UniClass(20,VL,'PtoDB')
# LM = [136,202,248,312,366]
# OP_uc = [[0.5369834,0.53468702,0.53629085,0.53497738,0.54233208],[0.57488471,0.57288283,0.57428098,0.57313596,0.58119461]]

for i in range(len(VL)):
    print(i)
    plt.scatter(np.log(LAD),np.log(OP_uc[i]),label = 'Vl=%.5f'%VL[i])
plt.legend()
plt.show()
'''
















#obj.phaseTs()
'''
    def phaseTs_critical(self):
        vr = [10]
        vl = np.linspace(-0.5653465839689737,-0.562,20)
        lm = [5,10]
        for x,y in list(it.product(vr,vl)):
            print(y)
            OP = []
            Nr = []
            for k in lm:
                obj = dm(J=1,Vr=x,Vl=y,Vlr=vlr,ladN=3,lm=k)
                lad = obj.Openladder_infl(3,k)
                _,_,SS,M,lad = obj.sweeps(1)
                Prob_r,Prob_l = DD.dimDen(lad,M,SS,x,y,vlr)
                l = len(lad)
                Nr.append(l)
                if lad[int(l/2)] == 'B':
                    #print(int(l/2)-5)
                    OP.append(Prob_l[int(l/2)-5])
                else:
                    j = self.closeB(lad,int(l/2))
                    #print(j,lad[j])
                    OP.append(Prob_l[j-5])
                print(OP)
                #DD.EET(lad,SS,x,y,vlr)
'''

'''
Columnar to II (along length of ladders)

(10,-0.5636842105263158)
[48,144,240]
[0.642807398210827, 0.6414192780965409, 0.6420499823243243]

(10,-0.5634210526315789)
[48,144,240]
[0.5950573150234831, 0.5734993274717773, 0.568608604651766]

(10,-0.5631578947368421)
[48,144,240]
[0.5446143972828185, 0.5150928116822542, 0.5135206774476515]
'''
'''
Columnar to II (along inflation of ladders)

-0.5653465839689737
sAssBssAssBssAssBssAssBssAssBssAssBssAssBssAssBs 48
sAssBssAsAsAsAssBssAssBssAsAsAsAssBssAssBssAsAsAsAssBssAssBssAsAsAsAssBssAssBssAsAsAsAssBssAssBssAsAsAsAssBssAssBssAsAsAsAssBssAssBssAsAsAsAssBs 144
sAssBssAsAsAsAssBssAssBssAssBssAssBssAssBssAsAsAsAssBssAssBssAsAsAsAssBssAssBssAssBssAssBssAssBssAsAsAsAssBssAssBssAsAsAsAssBssAssBssAssBssAssBssAssBssAsAsAsAssBssAssBssAsAsAsAssBssAssBssAssBssAssBssAssBssAsAsAsAssBssAssBssAsAsAsAssBssAssBssAssBssAssBssAssBssAsAsAsAssBssAssBssAsAsAsAssBssAssBssAssBssAssBssAssBssAsAsAsAssBssAssBssAsAsAsAssBssAssBssAssBssAssBssAssBssAsAsAsAssBssAssBssAsAsAsAssBssAssBssAssBssAssBssAssBssAsAsAsAssBs 432
[0.6195887180758808, 0.4023289031536746, 0.40239903512350533]

-0.5649943119722397
[0.601974807724817, 0.3897473769356247, 0.38978227887273004]

-0.5648181759738726
[0.5919294631753187, 0.38288037263124775, 0.3828957160152942]

-0.5646420399755055
[0.5809135331774729, 0.375613256905749, 0.3756149081056892]

-0.5641136319804045
[0.5422231491835292, 0.3512701340991717, 0.3512387024716806]
'''


