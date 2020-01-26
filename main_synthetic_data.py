#Author: Gautam Reddy Nallamala. Email: gautam_nallamala@fas.harvard.edu


import bass as md
import numpy as np
import sys

#Generate synthetic data. Some starting parameters are shown below. 
Sigma = 7
dict_size = 10
wordlength = 5
L = 5000 
#Input parameters. If eps = 0, need not specify Jthr.
Sigma = int(sys.argv[1]) #Alphabet size
dict_size = int(sys.argv[2]) #Dictionary size for synthetic data
wordlength = float(sys.argv[3]) #Mean motif length (Poisson distributed)
L = int(sys.argv[4]) #Size of dataset
eps_true = float(sys.argv[5]) #True e_p parameter
eps = float(sys.argv[6]) #e_p parameter used to build the dictionary. Action pattern noise. 
Jthr = float(sys.argv[7]) #The threshold on the JS divergence used during dictionary truncation. 
snr = float(sys.argv[8]) #The discriminability \mu defined in the paper. Syllable noise. 
p_b = float(sys.argv[9]) #The background noise e_b defined in the paper. This is the total probability of single letters from the alphabet. Background noise. 
trialnum = int(sys.argv[10]) #Just an index for the trial number. 

std = 1.0/snr
mu = 1.0

sys.stdout = open("test_S%d_D%d_l%d_L%d_epst%.2f_eps%.2f_Jthr%.2f_snr%.2f_p_b%.2f_%02d.dat"%(Sigma,dict_size,wordlength,L,eps_true,eps,Jthr,snr,p_b,trialnum), 'w')
#eps_true = 0.0


p_d_true = 0.5 #probability of a deletion given an error. 
p_ins_true = 0.2 #This paramter is unused since only one insertion is allowed. 
params_true = np.array([eps_true,p_d_true,p_ins_true,0,0,0,0,Sigma,std])

alphfreqs = np.random.dirichlet(5*np.ones(Sigma))
model_true = md.GMM_synthetic(params_true)
w_dict_true = md.generate_w_dict(alphfreqs,dict_size,wordlength)
P_w_true = np.zeros(len(w_dict_true))
P_w_true[:-Sigma] = np.random.dirichlet(1.0*np.ones(len(w_dict_true) - Sigma))*(1-p_b)
P_w_true[-Sigma:] = alphfreqs*p_b
Y,words_true,Y_data = md.generate_Y(L, P_w_true, w_dict_true, params_true,model_true)
lengths_Y = np.array([len(Y)],dtype = int)

np.save("Y_generated_S%d_D%d_l%d_L%d_epst%.2f_eps%.2f_Jthr%.2f_snr%.2f_p_b%.2f_%02d.npy"%(Sigma,dict_size,wordlength,L,eps_true,eps,Jthr,snr,p_b,trialnum), Y)
np.save("Y_data_generated_S%d_D%d_l%d_L%d_epst%.2f_eps%.2f_Jthr%.2f_snr%.2f_p_b%.2f_%02d.npy"%(Sigma,dict_size,wordlength,L,eps_true,eps,Jthr,snr,p_b,trialnum), Y_data)

#Solve
entropy = md.get_entropy(Y)
w_thr = 1e-5*np.exp(-entropy)
eps = 0.0
p_d = 0.5 
p_ins = 0.2 #Unused
mu = 1.0 #Unused. Was used to try priors on motif length.

H_beta_fac = 0 #Unused. Was used to introduce an additional entropic cost to the free energy. 
Sigma = 7 
Jthr = 0.3
params = np.array([eps,p_d,p_ins, mu, w_thr,H_beta_fac, Jthr, Sigma,std], dtype=float)
samps = np.random.choice(np.arange(len(Y)),len(Y),replace= False)
P_w, w_dict = md.solve_dictionary(Y,lengths_Y,params,model_true,15) #Solves for the dictionary. 

print("\n")
#Output
print("Dictionary")
md.print_dict(Y,w_dict,P_w)

print("\n")

print("\n")
#Output
print("True Dictionary")
md.print_dict(Y,w_dict_true,P_w_true)
print("\n")