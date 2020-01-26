import motifdiscovery as md
import numpy as np
import sys



#Generate data
eps = float(sys.argv[1])
p_d = float(sys.argv[2])
Jthr = float(sys.argv[3])
seed = int(sys.argv[4])
sys.stdout = open("motifs_acid_compare_eps%.2f_pd%.2f_Jthr%.2f_%d.dat" %(eps,p_d,Jthr,seed), 'w')
#eps_true = 0.0

data_explo_hmm = np.load("../Zebrafish_larvae/data_explo_hmm.npy")
data_dbsharppH_hmm = np.load("../Zebrafish_larvae/data_dbsharppH_hmm.npy")

lengths_explo_hmm = np.load("../Zebrafish_larvae/lengths_explo_hmm.npy")
lengths_dbsharppH_hmm = np.load("../Zebrafish_larvae/lengths_dbsharppH_hmm.npy")

model_fit = md.GMM_model(7)

means_ = np.load("../Zebrafish_larvae/acid_means.npy")
covars_ = np.load("../Zebrafish_larvae/acid_covars.npy")
weights_ = np.load("../Zebrafish_larvae/acid_weights.npy")
model_fit._read_params(means_,covars_,weights_)


np.set_printoptions(precision = 4, suppress = True)
print(np.argsort(model_fit.means_[:,0]))
print(model_fit.means_[np.argsort(model_fit.means_[:,0])])
print(model_fit.weights_[:,np.argsort(model_fit.means_[:,0])])

sys.stdout.flush()

lengths_dbsharppH = lengths_dbsharppH_hmm[:]
data_dbsharppH = data_dbsharppH_hmm[:np.sum(lengths_dbsharppH)]

HdbsharppH = -model_fit.score(data_dbsharppH,1)/len(data_dbsharppH) #entropy
YdbsharppH = np.exp(model_fit._compute_log_likelihood(data_dbsharppH))/np.exp(-HdbsharppH)

#Solve
w_thr = 1e-4
#eps = 0.0
#p_d = 0.5
p_ins = 0.2
mu = 1.0
H_beta_fac = 0
Sigma = YdbsharppH.shape[1]
std = 0.05
params = np.array([eps,p_d,p_ins, mu, w_thr,H_beta_fac, Jthr, Sigma, std], dtype =float)

P_w_dbsharppH, w_dict_dbsharppH = md.solve_dictionary(YdbsharppH,lengths_dbsharppH,params,model_fit,7)

print("\n")
#Output
print("Acid dictionary")
md.print_dict(YdbsharppH,w_dict_dbsharppH,P_w_dbsharppH)
sys.stdout.flush()

print("\n")

print("non-Markovianity: Acid")
transmat_, stationary_probs_ = md.compute_transmat(YdbsharppH)
a,b,c = md.test_for_markovianity(YdbsharppH,w_dict_dbsharppH,eps,p_d,transmat_, stationary_probs_)
sys.stdout.flush()

print("\n")



