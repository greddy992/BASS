#Author: Gautam Reddy Nallamala. Email: gautam_nallamala@fas.harvard.edu

#This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. 
#To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative Commons, 
#PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
cimport numpy as np
from libc.math cimport log
from libc.math cimport exp
from libc.math cimport ceil
from libc.math cimport pow
from libc.math cimport sqrt
from libc.math cimport cos
from libc.math cimport sin
from scipy.optimize import minimize
import scipy.stats as stats
from scipy.special import comb
from scipy.stats import poisson
import cython
import sys
from scipy.stats import norm
import editdistance
from copy import deepcopy

# cython: profile=True

#P_ygs is the character likelihood function q(y|.) in the paper. Outputs the probability of observing y given s. This is defined for the GMM_synthetic model. 
#This function specifies the emission probabilities of the mixture model. HAS to be defined.
@cython.cdivision(True)
cpdef double P_ygs(double [:] y, int s, int numclasses, double std):
    cdef int dim = 2
    cdef double means0 = 0
    cdef double means1 = 0
    cdef double PI = 3.1415926
    if s < numclasses - 1 and numclasses > 1:
        means0 =  cos(PI*2*s/(numclasses-1))
        means1 =  sin(PI*2*s/(numclasses-1))
    cdef double prob1 =  exp(-(y[0] - means0)*(y[0] - means0)/(2*std*std))/sqrt(2*PI*std*std)
    cdef double prob2 =  exp(-(y[1] - means1)*(y[1] - means1)/(2*std*std))/sqrt(2*PI*std*std)
    return prob1*prob2

#Sample from the mixture model. HAS to be defined. 
@cython.cdivision(True)
cpdef double [:] sample_y(int s, int numclasses, double std):
    cdef int dim = 2
    cdef double means0 = 0
    cdef double means1 = 0
    cdef double PI = 3.1415926
    if s < numclasses - 1:
        means0 =  cos(PI*2*s/(numclasses-1))
        means1 =  sin(PI*2*s/(numclasses-1))
        
    cdef double [:] Y = np.zeros(dim,dtype = float)
    Y[0] = means0 + std*np.random.randn()
    Y[1] = means1 + std*np.random.randn()
    return Y
    
#Q(Y|m) with no action pattern noise.     
cpdef double P_YgS_func_noeps(double [:,:] Y,  np.int_t[:] S):
    cdef int lenY = Y.shape[0]
    cdef int lenS = S.shape[0]
    cdef int i
    cdef double prod = 1
    if lenY != lenS:
        return 0
    else:
        for i in range(lenS):
            prod = prod*Y[i,S[i]]
        return prod
    
#This is the implementation of Q(Y|m) using the recursive formula from the paper.   
@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef double P_YgS_func(double[:,:] Y, np.int_t [:] S, double [:] params, double[:,:] P_memo):
    cdef int i,j
    cdef double eps,p_d,p_ins
    eps = params[0]
    p_d = params[1]
    p_ins = params[2]
    cdef int Ly = Y.shape[0]
    cdef int Ls = S.shape[0]
    cdef double P = 0
    if Ly == 0:
        return pow(eps*p_d,Ls)
    elif Ls == 0 or Ly < 0:
        return 0.0
    elif Ly > 2*Ls:
        return 0.0       
    elif P_memo[Ly-1,Ls-1] > -1:
        return P_memo[Ly-1,Ls-1]
    else:
        for i in range(min(3,Ly+1)):
            if i == 0 and eps*p_d > 1e-3:
                P += eps*p_d*P_YgS_func(Y[:],S[:Ls-1],params,P_memo)
            elif i == 1:
                P += (1-eps)*P_YgS_func(Y[:Ly-1],S[:Ls-1],params,P_memo)*Y[Ly-1,S[Ls-1]]
            elif i == 2 and eps*(1-p_d) > 1e-3: 
                P += eps*(1-p_d)*P_YgS_func(Y[:Ly-2],S[:Ls-1],params,P_memo)*Y[Ly-1,S[Ls-1]]*Y[Ly-2,S[Ls-1]]
#Can insert further cases here if more than insertion is to be allowed
    P_memo[Ly-1,Ls-1] = P
    return P


@cython.cdivision(True)
cpdef double P_YgS(double [:,:] Y,  np.int_t[:] S, double [:] params):
    cdef int lenY = Y.shape[0]
    cdef int lenS = S.shape[0]
    if params[0] < 1e-3 or lenS == 1:
        return P_YgS_func_noeps(Y,S)
    cdef double [:,::1] P_memo = -5*np.ones((lenY,lenS),dtype = float)
    return P_YgS_func(Y,S,params,P_memo)/(1.0 - pow(params[0]*params[1],lenS)) #divide by the possibility of S -> null.

#Maximum length of motif instantiations
def get_lmax(w_dict,params):
    eps = params[0]
    p_d = params[1]
    p_ins = params[2]
    lengths = [len(w) for w in w_dict]
    return 40#we set 20 as the length of maximal motif so 40 is the upper limit on the length of a motif instantiation with one insertion per symbol. 

#This function is part of the optimization of the code. Gives you the minimum and maximum lengths a motif can mutate to, based on the p_d and e_p. 
def get_lmin_and_max(w,params):
    l = len(w)
    eps = params[0]
    p_d = params[1]
    p_ins = params[2]
    lmin = 0
    lmax = 2*l
    for k in range(l):
        if comb(l,k)*pow(eps*p_d,k)*pow(1-eps,l-k) < 5e-3:
            lmin = l - k - 1
            break
    for k in range(l + 1):
        if comb(l,k)*pow(eps*(1-p_d),k)*pow(1-eps,l-k) < 5e-3:
            lmax = l + k
            break
    return lmin,lmax
    

#Append sequences
def append_mseq(sym,seqs):
    mseqs = []
    for i in range(len(seqs)):
        mseqs += [[sym] + seqs[i]]
    return mseqs

#Generate mutated sequences from a motif template m, drawn from P(\tilde{m}|m). 
def generate_mutated_sequences(seq,eps,p_d):
    mseqs = []
    probs_mseqs = []
    if len(seq) == 1:
        return [[seq[0]],[],[seq[0],seq[0]]],[1-eps,eps*p_d,eps*(1-p_d)]
    
    seqs,probs = generate_mutated_sequences(seq[1:],eps,p_d)
       
    mseqs += append_mseq(seq[0],seqs)
    probs_mseqs += [p*(1-eps) for p in probs]
    
    mseqs += seqs
    probs_mseqs += [p*eps*p_d for p in probs]
    
    seqs_dup = append_mseq(seq[0],seqs)
    seqs_dup = append_mseq(seq[0],seqs_dup)
    mseqs += seqs_dup
    probs_mseqs += [p*eps*(1-p_d) for p in probs]
    
    dups = []
    for i in range(len(mseqs)):
        for j in range(i+1,len(mseqs)):
            if np.array_equal(mseqs[i],mseqs[j]):
                dups += [j]
                probs_mseqs[i] += probs_mseqs[j]
                
    dups = np.array(dups)
    dups = np.unique(dups)                
    #print(dups)
    for j in range(len(dups)):
        del mseqs[dups[j]]
        del probs_mseqs[dups[j]]
        dups -= 1
        
    seqs_out = []
    probs_out = []
    for i in range(len(mseqs)):
        if probs_mseqs[i] > 5e-4:
            seqs_out += [mseqs[i]]
            probs_out += [probs_mseqs[i]]
            
    return seqs_out,probs_out

#Compute P(\tilde{m}|m) i.e., the probability of a mutated sequence given the motif template. 
def get_mutated_sequences_prob(seq,eps,p_d):
    if len(seq) == 1:
        return [seq],[1.0]
    else:
        seqs,probs = generate_mutated_sequences(seq,eps,p_d)
        empty_prob = 0
        for i in range(len(seqs)):
            if len(seqs[i]) == 0:
                empty_prob = probs[i]
                del seqs[i]
                del probs[i]
                break
            
        for i in range(len(seqs)):
            probs[i] /= (1-empty_prob)
        return seqs,probs


#This function computes the locations (index i) and lengths (index l) where a particular motif (index w) could possibly fit 
#based on a threshold w_thr_l on Q(Y_{i-l+1:i}|w). 
#Computing this is the rate-limiting step. 
#The MLE to get p_m is much more efficient if W_il is pre-computed and then optimization performed.
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def get_W_ils(w_dict, double[:,:] Y, np.int_t[:] lengths_Y, double [:] params):
    cdef int L = Y.shape[0]
    cdef int D = len(w_dict)
    cdef int K = lengths_Y.shape[0]
    cdef int lmax = get_lmax(w_dict,params)
    
    Q_gw = np.zeros((L,lmax), dtype=float) 
    cdef double[:,:] Q_gw_view = Q_gw
    
    cdef np.int_t[:] w_dict_view
    
    cdef int i,w,k,j,lmin,l,kk,mlen,wtild
    cdef double [:] Pmtilds
    cdef np.int_t [:] seq_mtild
    cdef double w_thr = params[4]
    cdef double w_thr_l = 0
    W_ils = []
    
    for w in range(D):
        W_ils += [[]]
        
        w_dict_view = w_dict[w]
        lmin,lmax = get_lmin_and_max(w_dict[w],params)
        lw = w_dict[w].shape[0]
        seqs,probs = get_mutated_sequences_prob(w_dict_view, params[0],params[1])
        Pmtilds = np.array(probs,dtype = float)
        mlen = Pmtilds.shape[0]
        
        Q_gw_view = np.zeros(Q_gw.shape, dtype=float) 
        for wtild in range(mlen):
            seq_mtild = np.array(seqs[wtild],dtype = int)
            l = seq_mtild.shape[0]-1
            kk = 0
            for k in range(K):
                L = lengths_Y[k]
                for i in range(l,L):
                    Q_gw_view[i+kk,l] += P_YgS_func_noeps(Y[i+kk-l:i+kk+1],seq_mtild)*Pmtilds[wtild]
                kk += L
        
        L = Y.shape[0]
        for i in range(L):
            for l in range(min(i+1,lmax)):
                w_thr_l = pow(w_thr,l+1)
                if Q_gw_view[i,l] > w_thr_l:
                    W_ils[w] += [[i,l,Q_gw_view[i,l]]] 
    return W_ils

#Compute the likelihood of a particular motif along the dataset. 
def get_Q_gw_from_W_il(W_ils_w,Y,w_dict,params):
    L = Y.shape[0]
    lmax = get_lmax(w_dict,params)
    Q_gw_w = np.zeros((L,lmax))
    for W_il in W_ils_w:
        Q_gw_w[<int> W_il[0],<int> W_il[1]] = W_il[2]
    return Q_gw_w


#Compute the likelihood of all motifs at a particular locus on the dataset. 
def get_Q_gw_i(double [:,:] Yseq, w_dict, double [:] params):
    cdef int L = Yseq.shape[0]
    cdef int lmax = get_lmax(w_dict,params)
    cdef int D = len(w_dict)
    Q_gw_i = np.zeros((lmax,D),dtype = float)
    cdef double [:,:] Q_gw_i_view = Q_gw_i
    cdef int l,w
    for l in range(min(L,lmax)):
        for w in range(D):
            Q_gw_i_view[l,w] = P_YgS(Yseq[-l-1:],w_dict[w],params)
    return Q_gw_i

#Evaluate the marginal probability Q(Y_{i-l+1:i}) at each locus i and length l. 
def evaluate_Q(double[:] P_w, W_ils, int L, int lmax, int D):
    Q = np.zeros((L,lmax), dtype=float)
    cdef double[:,:] Q_view = Q
    cdef int i,l,w,w_il_length
    
    for w in range(D):
        for W_il in W_ils[w]:
            i = <int> W_il[0]
            l = <int> W_il[1]
            Q_view[i,l] += P_w[w]*W_il[2]
    return Q

#Evaluate R defined in the paper. 
def evaluate_R(double[:,:] Q):
    cdef int L = Q.shape[0]
    cdef int i,l,lmax
    lmax = Q.shape[1]
    R = np.zeros(L, dtype = float)
    cdef double[:] R_view = R
    cdef double prod = 1
    for i in range(L):
        R_view[i] = Q[i,0] + 1e-10
        prod = 1
        for l in range(1,min(lmax,i+1)):
            prod *= R_view[i-l]
            R_view[i] += Q[i,l]/prod
    return R

#Evaluate R' defined in the paper. 
def evaluate_R1(double[:,:] Q):
    cdef int L = Q.shape[0]
    cdef int i,l,lmax
    lmax = Q.shape[1]
    R1 = np.zeros(L, dtype = float)
    cdef double[:] R1_view = R1
    cdef double prod = 1
    for i in range(L-1,-1,-1):
        R1_view[i] = Q[i,0] + 1e-10
        prod = 1
        for l in range(1,min(lmax,L-i)):
            prod *= R1_view[i+l]
            R1_view[i] += Q[i+l,l]/prod
    return R1

#Evaluate G defined in the paper. 
def evaluate_G(double [:] R,double [:] R1,int lmax):
    cdef int L = R.shape[0]
    G = np.zeros((L,lmax), dtype= float)
    cdef double[:,:] G_view = G
    cdef double prod = 1
    cdef double prod2 = 1
    cdef int i,l
    for i in range(L):
        prod *= R[i]/R1[i]
        prod2 = 1
        for l in range(min(lmax,i+1)):
            prod2 *= R[i-l]
            G_view[i,l] = prod/prod2
    return G

def evaluate_F(R):
    return -np.sum(np.log(R))


#Evaluate gradient of free energy
def evaluate_dF(double[:] P_w, double[:,:] G, W_ils):
    cdef int D = P_w.shape[0]
    
    dF = np.zeros(D, dtype = float)
    
    cdef double [::1] dF_view = dF
    
    cdef int i,l,w
    
    for w in range(D):
        for W_il in W_ils[w]:
            i = <int> W_il[0]
            l = <int> W_il[1]
            dF_view[w] += -G[i,l]*W_il[2] 
    return dF

#Evaluate free energy for a particular sequence. 
def evaluate_F_seq(Y,lengths_Y, P_w, w_dict, params):
    lmax = get_lmax(w_dict,params)
    L = Y.shape[0]
    D = len(w_dict)
    
    W_ils = get_W_ils(w_dict, Y,lengths_Y, params)
    Q = evaluate_Q(P_w,W_ils,L,lmax,D)
    R = evaluate_R(Q)
    F = evaluate_F(R)
    return F


#This is used in dictionary expansion. Compute the number of times the concatenated motif w1w2 occurs in the dataset upto a pre-factor. 
cpdef double subroutine_Nw1w2(np.int_t[:] w1w2, double [:,:] G, np.int_t [:] nz_pos, double [:,:] Q_gw1, double [:,:] Q_gw2, int l1,int l2, double [:] params):
    cdef int L = G.shape[0]
    cdef double Nw1w2 = 0
    cdef double eps = params[0]
    cdef double p_d = params[1]
    cdef double temp
    cdef int lenw1w2 = w1w2.shape[0]
    if lenw1w2 == 0:
        return 0.0
    cdef int i,l,lmin,lmax,k
    lmin,lmax = get_lmin_and_max(w1w2,params)
    for i in nz_pos:
        for l in range(lmin,min(lmax,i+1)):
            temp = 0
            temp += Q_gw2[i,l]*pow(eps*p_d,l1)
            temp += Q_gw1[i,l]*pow(eps*p_d,l2)
            for k in range(l):
                temp += Q_gw2[i,k]*Q_gw1[i-k-1,l-k-1]
            Nw1w2 += temp*G[i,l]
    return Nw1w2

#This is used to sample an output sequence Y given a motif template S. 
def sample(S,params,model):
    eps = params[0]
    p_d = params[1]
    p_ins = params[2]
    Y = []
    while len(Y) == 0:
        if len(S) == 1:
            Y += [model._generate_sample_from_state(S[0])]
        else:
            for s in S:
                randu = np.random.uniform()
                if randu < 1-eps:
                    Y += [model._generate_sample_from_state(s)]
                elif randu < 1-eps + eps*(1-p_d):
                    numy = 2#np.random.geometric(1-p_ins)
                    for j in range(numy):
                        Y += [model._generate_sample_from_state(s)]
    return Y

#Convert a data vector Y to a sequence of probability vectors q(Y_i|c_j).
def convert_Y_to_Y_Ps(Y,params,model):
    Sigma = int(params[7])
    Y_Ps = np.zeros((len(Y),Sigma),dtype = float)
    for i in range(len(Y)):
        for s in range(Sigma):
            Y_Ps[i,s] = model._compute_likelihood(Y[i],s)
    return Y_Ps

#Compute the Jensen-Shannon divergence matrix between all pairs of motifs in the dictionary. 
#This is where the editdistance library is used i.e.,  
#in order to restrict the computation to somewhat close motifs since most motifs are separated by the maximal distance 1.
#This takes annoyingly long to compute. 
@cython.cdivision(True)
def get_JS(w_dict,double [:] params, model):
    cdef int niter = 500
    cdef int D = len(w_dict)
    cdef double [:,:] dij = np.zeros((D,D),dtype = float)
    cdef double [:,:] dijT = np.zeros((D,D),dtype = float)
    cdef double [:,:] JS = np.zeros((D,D),dtype = float)
    
    eps = params[0]
    cdef np.int_t [:,:] editdist = np.zeros((D,D),dtype = int)
    cdef int di,dj,i
    cdef double dist
    for di in range(D):
        for dj in range(D):
            str1 = ''.join(str(e) for e in w_dict[di])
            str2 = ''.join(str(e) for e in w_dict[dj])
            dist = editdistance.eval(str1, str2)
            if dist > 0.0 and pow(eps,dist)*pow(1-eps,len(str1)-dist) < 1e-4:
                editdist[di][dj] = 1
                
    cdef double YiSi,YiSj
    cdef double [:,:] Ydi
    for di in range(D):
        for i in range(niter):
            sample_di = np.array(sample(w_dict[di],params,model),dtype = float)
            Ydi = convert_Y_to_Y_Ps(sample_di,params,model)
            YiSi = P_YgS(Ydi,w_dict[di],params)
            for dj in range(D):
                if w_dict[di].shape[0] == 1 or w_dict[dj].shape[0] == 1 or editdist[di][dj]:
                    dij[di,dj] = log(2)
                else:
                    YiSj = P_YgS(Ydi, w_dict[dj],params)
                    dij[di,dj] += (log(YiSi) - log(0.5*YiSi + 0.5*YiSj))/niter
    for di in range(D):
        for dj in range(D):
            JS[di][dj] = (dij[di,dj] + dij[dj,di])/(2*log(2))
    return JS

#This is used for the Markovianity test. Implements the forward algorithm for HMMs. 
cpdef double P_YgHMM(double[:,:] Y, double[:,:] transmat_hmm, double[:] rho):
    cdef int l = Y.shape[0]
    cdef int Sigma = Y.shape[1]
    alphas = np.zeros((l,Sigma),dtype = float)
    cdef double [:,:] alphas_view = alphas
    cdef double output = 0
    cdef int i,s,sp
    for i in range(l):
        for s in range(Sigma):
            if i == 0:
                alphas_view[i,s] = Y[i,s]*rho[s]
            else:
                for sp in range(Sigma):
                    alphas_view[i,s] += alphas_view[i-1,sp]*transmat_hmm[sp,s]
                alphas_view[i,s] *= Y[i,s]
    for s in range(Sigma):
        output += alphas_view[-1,s]
    return output

#This is used for the Markovianity test. Compute the expected counts from a Markovian model. 
def calculate_expected_frequency_hmm(seq,transmat,rho):
    l = len(seq)
    prob = 1
    for i in range(l):
        if i == 0:
            prob *= rho[seq[i]]
        else:
            prob *= transmat[seq[i-1]][seq[i]]
    return prob

#This is used for the Markovianity test. Empirical frequency of a motif. 
@cython.cdivision(True)
@cython.boundscheck(False)
cpdef double calculate_empirical_frequency_hmm(np.int_t [:] seq, double [:,:] Y, double [:,:] transmat_, double [:] stationary_probs_):
    cdef int l = seq.shape[0]
    cdef int L = Y.shape[0]
    cdef double prob = calculate_expected_frequency_hmm(seq,transmat_, stationary_probs_)
    cdef double counts = 0
    cdef int i,j
    cdef double count,likelihood
    for i in range(L-l):
        count = 1
        for j in range(l):
            count *= Y[i+j,seq[j]]
        likelihood = P_YgHMM(Y[i:i+l],transmat_, stationary_probs_)
        count /= likelihood
        count *= prob
        counts += count
    return counts/(L-l)

#Convert a list of motifs to a single array of letters. 
def create_word_array(ws,w_dict):
    arr = np.array([])
    for w in ws:
        arr = np.concatenate((arr,w_dict[w]))
    arr = np.array(arr,dtype = int)
    return arr

#This is zeta(m) from the paper. Computes the probability of all possible partitions of a motif. 
def Z_partitions(seq,P_w,w_dict):
    L = len(seq)
    D = len(w_dict)
    lengths = [len(w) for w in w_dict]
    lmax = np.max(lengths)
    Z_p = np.zeros(L)
    for i in range(L):
        for l in range(min(i+1,lmax)):
            for w in range(D):
                if np.array_equal(seq[i-l:i+1],w_dict[w]):
                    if i >= l+1:
                        Z_p[i] += Z_p[i-l-1]*P_w[w]
                    else:
                        Z_p[i] += P_w[w]
    return Z_p[-1] 

#Computes the number of times a concatenated motif occurs in the dataset. The subroutine_Nw1w2 is called here. 
#Directly gives the p-value for the over-representation of the concatenated motif relative to random juxtaposition. 
def Nw1w2(ws, P_w, w_dict, Y, Q, Q_gw1, Q_gw2, nz_pos, params):
    arr = create_word_array(ws,w_dict)
    lengths = [len(w) for w in w_dict]
    l1 = len(w_dict[ws[0]])
    l2 = len(w_dict[ws[1]])
    
    lmax = get_lmax([arr],params)
    
    R = evaluate_R(Q)
    R1 = evaluate_R1(Q)
    G = evaluate_G(R,R1,lmax)
    
    Fseq = -np.log(Z_partitions(arr,P_w,w_dict))
    
    N_calc = np.exp(-Fseq)*subroutine_Nw1w2(arr,G,nz_pos,Q_gw1,Q_gw2,l1,l2,params) + 1e-10
    f_calc = N_calc/len(Y)
    if N_calc < 5.0:
        return 1.0
    
    lmean = np.sum(lengths*P_w)
    N_exp = len(Y)*np.exp(-Fseq)/lmean + 1e-10
    f_exp = N_exp/len(Y)
    
    q1 = 1 + (1.0/f_exp + 1.0/(1-f_exp) - 1)/(6.0*len(Y)) #correction to LR test
    m2lnLR = 2*len(Y)*(f_calc*np.log(f_calc/f_exp) + (1-f_calc)*np.log((1-f_calc)/(1-f_exp)))/q1
    return stats.chi2.sf(m2lnLR,1)


#Free energy as a function of betas. beta_i = log(p_{-1}/p_i). 
def F_beta(betas, W_ils, L, lmax, D, params):
    P_w = np.zeros(len(betas)+1)
    P_w[:-1] = np.exp(-betas)
    P_w[-1] = 1.0
    P_w /= np.sum(P_w)
    for i in range(len(P_w)):
        if P_w[i] != P_w[i]:
            print("Error in F_beta", i, betas[i])
            sys.stdout.flush()
    Q = evaluate_Q(P_w, W_ils, L, lmax, D)
    R = evaluate_R(Q)
    F = evaluate_F(R)
    return F/L

#Gradient of free energy in terms of betas.                                     
def dF_beta(betas, W_ils, L, lmax, D,params):
    P_w = np.zeros(D)
    P_w[:-1] = np.exp(-betas)
    P_w[-1] = 1.0
    P_w /= np.sum(P_w)
    
    Q = evaluate_Q(P_w, W_ils, L, lmax, D)
    R = evaluate_R(Q)
    R1 = evaluate_R1(Q)
    
    G = evaluate_G(R,R1,lmax)
    
    dF_Pw = evaluate_dF(P_w, G, W_ils)
    
    dF_betas = np.zeros(len(betas))
    for j in range(D-1):
        dF_betas[j] = P_w[j]*(-dF_Pw[j] + np.sum(P_w*dF_Pw))
        
    return dF_betas/L

#Minimizing free energy using gradient descent. 
def minimize_F(Y, W_ils, L, lmax, D, params, method):
    betas0 = 0.1*np.random.randn(D-1)
    bounds = []
    for i in range(len(betas0)):
        bounds += [[-20.0,20.0]]
    res = minimize(F_beta, betas0, args = (W_ils, L, lmax, D, params),method = method, jac = dF_beta, bounds = bounds, options = {'disp':False})
    return res


#Optimizing for P_w. MLE. 
def get_P_w(Y, lengths_Y, w_dict, params,method = 'L-BFGS-B'):
    lmax = get_lmax(w_dict, params)
    L = len(Y)
    D = len(w_dict)
    W_ils = get_W_ils(w_dict, Y, lengths_Y, params)
    res = minimize_F(Y,W_ils, L, lmax, D,params,method) 
    P_w_fit = np.zeros(D)
    P_w_fit[:-1] = np.exp(-res.x)
    P_w_fit[-1] = 1.0
    P_w_fit /= np.sum(P_w_fit)
    return P_w_fit

#This is called in the "decode" function below. 
def decode_Y(Y,P_w,w_dict,params):
    D = len(w_dict)
    L = len(Y)
    K = np.zeros(L)
    lmax = get_lmax(w_dict,params)
    decodedws = []
    decodedls = []
    argmaxws = np.zeros(L)
    argmaxls = np.zeros(L)
    for i in range(L):
        l = min(i,lmax-1)
        Q_gw_i = get_Q_gw_i(Y[i-l:i+1],w_dict,params)
        if i == 0:
            K[i] = np.log(np.max(P_w*Q_gw_i[0]))
            argmaxws[0] = np.argmax(P_w*Q_gw_i[0])
            argmaxls[0] = 0
        else:
            argmaxw = np.argmax(P_w*Q_gw_i,axis=1)
            maxw = np.max(P_w*Q_gw_i,axis=1)
            ls = []
            for l in range(min(i+1,lmax)):
                if l == i:
                    ls += [np.log(maxw[l])]
                else:
                    ls += [np.log(maxw[l]) + K[i-l-1]]
            argmaxl = np.argmax(ls)
            K[i] = np.max(ls)
            argmaxws[i] = argmaxw[argmaxl]
            argmaxls[i] = argmaxl
    i = L-1
    while i >= 0:
        decodedws += [int(argmaxws[int(i)])]
        decodedls += [int(argmaxls[int(i)])+1]
        i -= argmaxls[int(i)]+1
    ws = np.array(decodedws[::-1])
    ls = np.array(decodedls[::-1])
    
    w_ML = []
    for w in ws:
        w_ML += list(w_dict[w])
    
    return w_ML,ws,ls

#Decoding the most likely sequence of motifs given the dataset and the dictionary of motifs using the Viterbi-like algorithm.      
def decode(Y,lengths_Y,w_dict,params):
    L = len(Y)
    D = len(w_dict)
    lmax = get_lmax(w_dict,params)
    P_w = get_P_w(Y, lengths_Y, w_dict, params)
    
    kk = 0
    w_MLs = []
    words = []
    wordlengths = []
    for lths in lengths_Y:
        w_ML,ws,ls = decode_Y(Y[kk:kk+lths],P_w,w_dict,params)
        w_MLs += [w_ML]
        words += [ws]
        wordlengths += [ls]
        kk += lths
    return w_MLs,words,wordlengths

#Generating synthetic data
def generate_Y(L, P_w, w_dict, params, model):
    l = 0
    Y = []
    ws = []
    words_true = []
    while l < L:
        w = np.random.choice(len(P_w), 1, p=P_w)[0]
        ws += list(w_dict[w])
        words_true += [w]
        S = sample(w_dict[w],params,model)
        Y = Y + S
        l += len(S)
    Ydata = np.array(Y)
    return convert_Y_to_Y_Ps(Y,params,model),words_true,Ydata
    
#Generating a synthetic dictionary    
def generate_w_dict(alphfreqs, D, lmean):
    w_dict= []
    alphabetsize = len(alphfreqs)
    flag = 0
    while len(w_dict) < D:
        numletters = poisson.rvs(lmean)#np.random.geometric(1.0/lmean)#
        while numletters < 2:
            numletters = poisson.rvs(lmean)#np.random.geometric(1.0/lmean)
        w = np.zeros(numletters)
        for j in range(numletters):
            w[j] = np.random.choice(alphabetsize, 1, p=alphfreqs)[0]
        
        for ww in w_dict:
            if len(w) == len(ww) and np.prod((w == ww)):
                flag = 1
        if flag == 1:
            flag = 0
            continue
        w_dict += [w]
    for i in range(alphabetsize):
        w_dict += [np.array([i])]
    w_dict = [np.array(w, dtype = int) for w in w_dict]
    return w_dict

#Removing duplicate motifs from the dictionary. This is the function that implements dictionary truncation based on JS divergence. 
def remove_duplicates_w_dict(P_w,w_dict,params,model):
    eps = params[0]
    dups = []
    if eps > 1e-3:
        JS = get_JS(w_dict,params,model)
        Jthr = params[6]
        P_w_weighted = np.sum((JS < Jthr)*P_w,axis=1)
        
        for i in range(len(w_dict)):#Remove duplicates
            for j in range(i+1,len(w_dict)): 
                if JS[i][j] < Jthr and P_w_weighted[i] >= P_w_weighted[j]:
                    dups += [j]
                elif JS[i][j] < Jthr and P_w_weighted[i] < P_w_weighted[j]:
                    dups += [i]
                    break
    else:
        for i in range(len(w_dict)):#Remove duplicates
            for j in range(i+1,len(w_dict)): 
                if np.array_equal(w_dict[i],w_dict[j]):
                    dups += [j]
        
    dups = np.array(dups)
    dups = np.unique(dups)                
    print(dups)
    for j in range(len(dups)):
        del w_dict[dups[j]]
        dups -= 1
    return w_dict

#Further truncate dictionary if the motif occurs fewer than a certain number of times. 5 copies is the currently set threshold
def truncate_w_dict(P_w,w_dict,thr = 5e-4): 
    i = 0
    P_w = list(P_w)
    while i < len(w_dict):#Remove low probability words
        if P_w[i] < thr and len(w_dict[i]) > 1:
            del w_dict[i]
            del P_w[i]
            i-=1
        i+=1
    return w_dict

#Keep truncating until all motifs occurs more than 5 times. In its current implementation only truncates once. 
def prune_w_dict(Y,lengths_Y, w_dict, params,model):
    D = len(w_dict)
    Dnew = 0
    i=0
    P_w = get_P_w(Y,lengths_Y, w_dict, params)
    w_dict = remove_duplicates_w_dict(P_w,w_dict,params,model)

    while Dnew != D and i < 1:
        D = len(w_dict)
        P_w = get_P_w(Y,lengths_Y, w_dict, params)
        lengths = [len(w) for w in w_dict]
        lmean = np.sum(lengths*P_w)
        thr = 5*lmean/len(Y)  #Threshold of 5 copies is set here. 
        w_dict = truncate_w_dict(P_w,w_dict,thr)
        Dnew = len(w_dict)
        i+=1
    return w_dict

#This is used at the very end of the algorithm. Used to remove low probability single letters from the dictionary. 5 copies is the threshold again.
def prune_letters_w_dict(Y,P_w,w_dict): 
    lengths = [len(w) for w in w_dict]
    lmean = np.sum(lengths*P_w)
    thr = 5*lmean/len(Y)
    
    P_w = list(P_w)
    i = 0
    while i < len(w_dict):#Remove low probability words
        if P_w[i] < thr and len(w_dict[i]) == 1: 
            del w_dict[i]
            del P_w[i]
            i-=1
        i+=1
    return w_dict

#This is the main function used for dictionary expansion. This function is heavily optimized. 
def get_words_to_add(Y,lengths_Y,w_dict,params):
    words_to_add = []
    P_w = get_P_w(Y,lengths_Y, w_dict, params)
    lengths = [len(w) for w in w_dict]
    lmax_w_dict = np.max(lengths)
    L = len(Y)
    lmax = get_lmax(w_dict,params)
    D = len(w_dict)
    W_ils = get_W_ils(w_dict, Y,lengths_Y, params)
    Q = evaluate_Q(P_w, W_ils, L, lmax, D)
    
    R = evaluate_R(Q)
    R1 = evaluate_R1(Q)
    G = evaluate_G(R,R1,lmax)
    w_thr = params[4]
    lmean = np.sum(P_w*lengths)
    for ibeta in range(len(w_dict)):
        lmin,lmax = get_lmin_and_max(w_dict[ibeta],params)
        Nw1w2_arr = np.zeros(len(w_dict))
        Q_gw2 = get_Q_gw_from_W_il(W_ils[ibeta],Y,w_dict,params)
        
        nz_pos = np.nonzero(np.sum(Q_gw2[:,lmin:lmax],axis=1) > w_thr)
        nz_pos = np.array(nz_pos[0],dtype = int)
        
        for ialpha in range(len(w_dict)):
            ws = [ialpha,ibeta]
            arr = create_word_array(ws,w_dict)
            Q_gw1 = get_Q_gw_from_W_il(W_ils[ialpha],Y,w_dict,params)
            if len(arr) > 25: #Not sure this is required. This is the upper limit on the length of motifs. 
                continue
            pvalue_ij = Nw1w2(ws,P_w,w_dict,Y,Q,Q_gw1,Q_gw2,nz_pos, params)
            if pvalue_ij < 1e-3:  #p_value of over-representation of concatenated motif should be less than 0.001 to accept a new motif to the dictionary. 
                words_to_add += [arr]
                
    return words_to_add

#One iteration of dictionary expansion and truncation.
def update_w_dict(Y,lengths_Y,w_dict,params,model):
    words_to_add = get_words_to_add(Y,lengths_Y,w_dict,params) #expand              
    w_dict = w_dict + words_to_add
    print("Dictionary length %d" %len(w_dict))
    w_dict = prune_w_dict(Y,lengths_Y,w_dict,params,model) #truncate
    print("Pruned length %d" %len(w_dict))
    sys.stdout.flush()
    return w_dict

def get_entropy(Y):
    P_S = np.zeros(Y.shape[1])
    P_Y = np.sum(Y,axis=1)
    P_Si = np.mean(Y/P_Y[:,np.newaxis],axis=0)
    P_Yi = np.sum(Y*P_Si,axis=1)
    entropy = np.mean(-P_Yi*np.log(P_Yi+1e-15))
    return entropy

#Main function used to run the algorithm. 
#Takes in sequences of probability vectors over an alphabet of size Sigma and the lengths of each sequence (the sum of lengths_Y should equal total length of Y)
#Returns the probabilities of each motif and the dictionary itself. 
def solve_dictionary(Y,lengths_Y,params,model,niter = 6):
    if np.sum(lengths_Y) != len(Y) or np.min(lengths_Y) <= 0:
        print("Invalid lengths_Y", np.sum(lengths_Y), len(Y))
        sys.stdout.flush()
        return 
    
    w_dict = []
    Sigma = Y.shape[1]
    params[7] = Sigma

    for i in range(Sigma):
        w_dict += [np.array([i], dtype = int)]
        
    P_w = get_P_w(Y,lengths_Y,w_dict,params)
    
    Fs = np.zeros(niter+1)
    Fs[0] = evaluate_F_seq(Y,lengths_Y,P_w, w_dict, params)/len(Y)
    
    w_dict = prune_w_dict(Y,lengths_Y,w_dict,params,model)
    
    for i in range(niter):
        w_dict = update_w_dict(Y,lengths_Y,w_dict,params,model)
        P_w = get_P_w(Y,lengths_Y,w_dict,params)
        Ftrain = evaluate_F_seq(Y, lengths_Y,P_w, w_dict, params)/len(Y)
        Fs[i+1] = Ftrain 
        w_dict_list = [list(w) for w in w_dict]
        print("%d iter, w_dict length = %d, Train -logL = %.3f" %(i+1,len(w_dict),Ftrain))
        sys.stdout.flush()
        if i == niter - 1 or (i > 1 and abs(Fs[i] - Fs[i+1]) < 0.002 and abs(Fs[i-1] - Fs[i]) < 0.002):
            w_dict = prune_letters_w_dict(Y,P_w,w_dict)
            w_dict = prune_w_dict(Y,lengths_Y,w_dict,params,model)
            P_w = get_P_w(Y, lengths_Y, w_dict, params)
            break
    print("Final length of w_dict = %d" %(len(w_dict)))
    print("Done, w_dict length = %d" %(len(w_dict)))
    return P_w,w_dict

#Definition of the model. This is important. You need to define your own P_ygs and sample_y. The current implementation of these two functions is for 
#the synthetic data case presented in the paper. 
class GMM_synthetic:
    def __init__(self,params):
        self.Sigma = int(params[7])
        self.std = float(params[8])
    def _compute_likelihood(self,y,s):
        return P_ygs(y,s,self.Sigma,self.std)
    def _generate_sample_from_state(self,s):
        return sample_y(s,self.Sigma,self.std)

#This is our implementation of a GMM used to fit multiple datasets simultaneously (see paper). Not used for the synthetic dataset.
class GMM_model:
    def __init__(self,numclasses):
        self.numclasses = numclasses
        
    def E_step(self,datasets):
        N = datasets.shape[1]
        numsets = datasets.shape[0]
        gamma_ = np.zeros((numsets,N,self.numclasses))
        for k in range(self.numclasses):
            gamma_[:,:,k] = 1e-20 + self.weights_[:,k][:,np.newaxis]*stats.multivariate_normal.pdf(datasets,mean = self.means_[k],cov = self.covars_[k])
        gamma_ = gamma_/np.sum(gamma_,axis=2)[:,:,np.newaxis]
        return gamma_
    
    def M_step(self,datasets,gamma_):
        for k in range(self.numclasses):
            Nk = np.sum(gamma_[:,:,k])
            self.means_[k] = np.sum(np.sum(gamma_[:,:,k][:,:,None]*datasets,axis=1),axis=0)/Nk
            outerprod = (datasets - self.means_[k])[:,:,:,None]*(datasets - self.means_[k])[:,:,None,:]
            self.covars_[k] = np.sum(np.sum(gamma_[:,:,k][:,:,None,None]*outerprod,axis=1),axis=0)/Nk
            self.weights_ = np.sum(gamma_,axis=1)/self.N
            
    def LL(self,datasets):
        N = datasets.shape[1]
        numsets = datasets.shape[0]
        temp = np.zeros((numsets,N))
        for k in range(self.numclasses):
            temp += self.weights_[:,k][:,None]*stats.multivariate_normal.pdf(datasets,mean = self.means_[k],cov = self.covars_[k])
        LL = np.mean(np.log(temp + 1e-80))
        return -LL
        
    def solve(self,datasets):
        self.numsets= len(datasets)
        self.dim = datasets.shape[2]
        self.N = datasets.shape[1]
        self.means_ = np.zeros((self.numclasses,self.dim))
        self.covars_ = np.zeros((self.numclasses, self.dim,self.dim))
        self.weights_ = np.zeros((self.numsets,self.numclasses))
        
        datasets_flat = np.reshape(datasets,(-1,datasets.shape[2]))
        covar = np.cov(datasets_flat, rowvar = False)
        mean = np.mean(datasets_flat, axis = 0)
        
        numinits = 20
        means_init = np.zeros((numinits,self.numclasses,self.dim))
        covars_init = np.zeros((numinits,self.numclasses,self.dim,self.dim))
        weights_init = np.zeros((numinits,self.numsets,self.numclasses))
        LL_init = np.zeros(numinits)
        for init_ in range(numinits):
            for i in range(self.numclasses):
                means_init[init_][i] = np.random.multivariate_normal(mean,covar)
                covars_init[init_][i] = deepcopy(covar)

            for j in range(self.numsets):
                weights_init[init_][j] = np.random.dirichlet(5*np.ones(self.numclasses))
            self.means_ = means_init[init_]
            self.covars_ = covars_init[init_]
            self.weights_ = weights_init[init_]
            LL_init[init_] = self.LL(datasets)
        best = np.argmin(LL_init)
        self.means_ = means_init[best]
        self.covars_ = covars_init[best]
        self.weights_ = weights_init[best]
            
        LL_curr = self.LL(datasets)
        LL_prev = 0
        print("Initial negative log-likelihood per sample = %.4f" %LL_curr)
        num = 0
        while np.abs(LL_curr - LL_prev) > 1e-4:
            gamma_= self.E_step(datasets)
            self.M_step(datasets,gamma_)
            LL_prev = LL_curr
            LL_curr = self.LL(datasets)
            num += 1
            #print(LL_curr)
        print("Final negative log-likelihood per sample = %.4f" %LL_curr)
        print("Number of iterations = %d" %num)
        
    def _compute_posterior(self,y,set_index):
        post = np.zeros(self.numclasses)
        for k in range(self.numclasses):
            post[k] = self.weights_[set_index][k]*self._compute_likelihood(y,k)
        return post/np.sum(post)
        
    def _compute_likelihood(self,y,s):
        return stats.multivariate_normal.pdf(y,mean = self.means_[s],cov = self.covars_[s])
    
    def _compute_log_likelihood(self,data):
        Y = np.zeros((len(data),self.numclasses))
        for k in range(self.numclasses):
            Y[:,k] = np.log(stats.multivariate_normal.pdf(data,mean = self.means_[k],cov = self.covars_[k]) + 1e-80)
        return Y
    def score(self,dataset,set_index):
        temp = np.zeros(len(dataset))
        for k in range(self.numclasses):
            temp += self.weights_[set_index,k]*stats.multivariate_normal.pdf(dataset,mean = self.means_[k],cov = self.covars_[k])
        LL = np.sum(np.log(temp + 1e-80))
        return LL
        
    def _generate_sample_from_state(self,s):
        return np.random.multivariate_normal(self.means_[s],self.covars_[s])
    
    def _read_params(self,means_,covars_,weights_):
        self.numclasses = means_.shape[0]
        self.means_ = means_
        self.covars_ = covars_
        self.weights_ = weights_
        
    def _save_params(self,filename):
        np.save(filename + "_means",self.means_)
        np.save(filename + "_covars",self.covars_)
        np.save(filename + "_weights",self.weights_)


#The functions below are for the various tests and comparisons done in the paper. 
        
#Implementation of the forward backward algorithm to compute the transition matrix for the HMM.         
def compute_transmat(Y):
    #Initialize transition matrices
    numclasses = Y.shape[1]
    transmat_ = np.zeros((numclasses,numclasses))
    stationary_probs_ = np.random.dirichlet(5*np.ones(numclasses))
    for k in range(numclasses):
        transmat_[k] = np.random.dirichlet(5*np.ones(numclasses))
    for k in range(numclasses):
        for iter_ in range(50):
            stationary_probs_ = np.einsum('i,ij', stationary_probs_, transmat_)
            
    numiter = 100
    LLs = np.zeros(numiter)
    
    #compute alphas
    for iter_ in range(numiter):
        N = Y.shape[0]
        alphas = np.zeros((N,numclasses))
        norms = np.zeros(N)
        for i in range(N):
            if i == 0:
                alphas[i] = stationary_probs_*Y[i] + 1e-12
            else:  
                alphas[i] = np.einsum('i,ij',alphas[i-1],transmat_)*Y[i] + 1e-12
            norms[i] = np.sum(alphas[i])
            alphas[i] /= norms[i]

        
        LLs[iter_] = -np.sum(np.log(norms + 1e-20))/N
        #print(iter_,LLs[iter_])
        if iter_ > 5 and np.abs(LLs[iter_] - LLs[iter_-1]) < 1e-4:
            break
        #compute betas
        betas = np.zeros((N,numclasses))
        for i in range(N-1,-1,-1):
            if i == N-1:
                betas[i] = 1.0 + 1e-12
            else:  
                betas[i] = np.einsum('j,ij,j', betas[i+1], transmat_,Y[i+1]) + 1e-12
            betas[i] /= np.sum(betas[i])

        #compute states
        gamma_ = np.zeros((N,numclasses))
        for i in range(N):
            gamma_[i] = alphas[i]*betas[i]/np.sum(alphas[i]*betas[i])

        #compute transitions:
        xi_ = np.zeros((N,numclasses,numclasses))
        for j in range(numclasses):
            for k in range(numclasses):
                xi_[1:,j,k] = alphas[:-1,j]*transmat_[j,k]*Y[1:,k]*betas[1:,k]
        xi_ /= np.sum(xi_,axis=(1,2))[:,None,None]

        #update transmat_
        for k in range(numclasses):
            transmat_[:,k] = np.sum(xi_[1:,:,k],axis=0)/np.sum(gamma_[:-1],axis=0)
        stationary_probs_ = gamma_[0]
        for k in range(numclasses):
            transmat_[k] /= np.sum(transmat_[k])
            
        for k in range(numclasses):
            for iter_ in range(50):
                stationary_probs_ = np.einsum('i,ij', stationary_probs_, transmat_)
    
    return transmat_,stationary_probs_

#Main function used to test markovianity of each motif. 
def test_for_markovianity(Y,w_dict,eps,p_d,transmat_, stationary_probs_):
    lengths = [len(w) for w in w_dict]
    lmean = np.mean(lengths)
    mlnPs = np.zeros(len(w_dict))
    emps =  np.zeros(len(w_dict))
    exps =  np.zeros(len(w_dict))
    for i,w in enumerate(w_dict):
        seqs,probs = get_mutated_sequences_prob(list(w),eps,p_d)
        emp = 0
        exp = 0
        for j,seq in enumerate(seqs):
            seq_arr = np.array(seq,dtype = int)
            #print(w,seq_arr,probs[i])
            emp += calculate_empirical_frequency_hmm(seq_arr,Y,transmat_, stationary_probs_)*probs[j]
            exp += calculate_expected_frequency_hmm(seq_arr,transmat_, stationary_probs_)*probs[j]

        q1 = 1 + (1.0/exp + 1.0/(1-exp) - 1)/(6.0*len(Y)) #correction to LR test
        ll = 2*len(Y)*(emp*np.log(emp/exp) + (1-emp)*np.log((1-emp)/(1-exp)))/q1
        mlnP = -np.log10(stats.chi2.sf(ll,1))
        mlnPs[i] = mlnP
        emps[i] = emp
        exps[i] = exp
        #print("%04d %04d %2.2f"%(int(emp*len(Y)),int(exp*len(Y)),mlnP),w)
    sorted_ = np.argsort(-mlnPs)
    for w in sorted_:
        if emps[w] > exps[w] and 10**(-mlnPs[w]) < 1e-3:
            print("%04d %04d %2.2f"%(int(emps[w]*len(Y)),int(exps[w]*len(Y)),mlnPs[w]),w_dict[w])
    return mlnPs,emps,exps

#Print dictionary
def print_dict(Y,w_dict,P_w):
    sorted_ = np.argsort(-P_w)
    lengths = [len(w) for w in w_dict]
    lmean = np.mean(lengths)
    for i in sorted_[:]:
        print("%.4f %d"%(P_w[i],int(P_w[i]*len(Y)/lmean)),w_dict[i])

#Combine two dictionaries:
def combine_dicts(w_dict1, w_dict2, params, model):
    w_dict = w_dict1 + w_dict2
    eps = params[0]
    params[0] = 0
    P_w = []
    w_dict = remove_duplicates_w_dict(P_w,w_dict,params,model)
    return w_dict

#Compare the number of occurrences of each motif in two datasets. The model of course has to be the same for both datasets.     
def compare_datasets(Y1, lengths_Y1, Y2, lengths_Y2,  w_dict1, w_dict2, params,model):
    w_dict = combine_dicts(w_dict1,w_dict2,params,model)
    P_w1 = get_P_w(Y1,lengths_Y1,w_dict,params)
    P_w2 = get_P_w(Y2,lengths_Y2,w_dict,params)
    lengths = [len(w) for w in w_dict]
    lmean = np.sum(P_w2*lengths)
    N_av2 = len(Y2)/lmean
    scores = np.zeros(len(w_dict))
    print(len(w_dict), len(w_dict1), len(w_dict2))
    emps = np.zeros(len(w_dict))
    exps = np.zeros(len(w_dict))
    for w in range(len(w_dict)):
        f_calc = P_w2[w]
        f_exp =  P_w1[w]
        q1 = 1 + (1.0/f_exp + 1.0/(1-f_exp) - 1)/(6.0*N_av2) #correction to LR test
        m2lnLR = 2*N_av2*(f_calc*np.log(f_calc/f_exp) + (1-f_calc)*np.log((1-f_calc)/(1-f_exp)))
        scores[w] = -np.log10(stats.chi2.sf(m2lnLR,1))
        emps[w] = f_calc*N_av2
        exps[w] = f_exp*N_av2
        
    sorted_ = np.argsort(-scores)
    for w in sorted_:
        if P_w2[w] > P_w1[w] and N_av2*P_w2[w] > 10 and len(w_dict[w]) > 1 and 10**(-scores[w]) < 1e-2 and N_av2*P_w1[w] > 5:
            print( "%04d %04d %.2f" %(int(N_av2*P_w1[w]),int(N_av2*P_w2[w]), scores[w]),w_dict[w])
    return scores,emps,exps
    


