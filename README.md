# BASS
Code for the "Behavioral Action Sequence Segmentation" algorithm. 

This folder contains all the code for analysis and the implementation of the motif discovery algorithm BASS and instructions to reproduce the figures in the paper titled "A lexical approach for identifying behavioral action sequences". 

The full data is required to reproduce the figures from the papers is uploaded separately due to its size. See the paper for a link to the dataset. This dataset is not required to use BASS. 

If your goal is to simply use BASS, reading the first section of this README will suffice. 

For questions, email gautam_nallamalaATfas.harvard.edu

Behavioral Action Sequence Segmentation (BASS) usage:
Three files are used for BASS:
bass.pyx
main_synthetic_dataset.py
setup_bass.py

To run the algorithm, Python3, Cython, NumPy, SciPy are required. editdistance is also required, which can be installed using the command:

pip3 install editdistance

Before running any of the code, bass.pyx has to be compiled using:

python3 setup_bass.py build_ext --inplace 

To run a test case, use:

python3 main_synthetic_data.py 7 50 5 10000 0.0 0.0 0.15 4 0.0 1 &

bass.pyx contains the implementation of the motif discovery algorithm, the specification of the mixture model and miscellaneous functions used for the analysis in the paper.  

main_synthetic_dataset.py is a sample application of the algorithm. This code generates a synthetic dictionary of motifs, a dataset from that dictionary and applies the motif discovery algorithm on the dataset. The output files contain the true dictionary and the one learned by the algorithm along with the probabilities of each motif. 

Important:
For each new application, a `soft’ clustering model has to be specified. In the code, this is implemented in the GMM_synthetic class in bass.pyx. This class has to be appropriately modified or alternatively a new class should be defined which contains the two functions defined for this class – “_compute_likelihood” and “_generate_sample_from_state”. 


Analysis of tracked zebrafish larvae data:

Raw data is in the four folders
resultsMay2019
Catamaran_pH_2bTxtOnly
Catamaran_pH_2cTxtOnly
resultsForSB1

The code used for analysis of the tracked fish data is included in the Jupyter notebook titled “Zebrafish_larvae_analysis_acid_data_final.ipynb”. This includes the extraction of the tracking data, computing the various parameters for each bout – speed, delta heading, duration of bout, tail amplitude, etc. Also includes clustering and generation of various plots presented in the paper. 

The three files named acid_*.npy contain the mixture model parameters presented in the paper.

The files named “data_explo_hmm.npy”, “data_dbsharppH_hmm.npy”, “lengths_explo_hmm.npy” and “lengths_dbsharppH_hmm.npy” are reduced versions of the dataset containing only the information used for clustering and BASS. 

Using BASS to reproduce results from pH experiments:

The Python code to run BASS on the exploration and aversive environment data is in 
main_explo.py
main_acid.py

These take input the Gaussian Mixture Model learned in the previous section, whose parameters are stored in the three files named acid_*.npy. They also require the four data files “data_explo_hmm.npy”, “data_dbsharppH_hmm.npy”, “lengths_explo_hmm.npy” and “lengths_dbsharppH_hmm.npy” generated from the analysis.

The parameters used in the paper are eps = 0.1, p_d = 0.2, Jthr = 0.15. To start with, we recommend running these two files with parameter eps = 0 since it is quicker to converge. The rest of the parameters are unused except for w_thr (anything less than 1e-3 works equally well, setting too low takes very long time to converge). 

The output from BASS can be read by another Jupyter notebook, see below. 

Analysis of output from BASS:

The code used to analyze the output from BASS is included in “analyze BASS output files, final version.ipynb”. This includes extracting the dictionary of motifs in exploratory and aversive environments and generating the various plots presented in the paper. 

Some of the files generated in this code are used as input for the “Zebrafish_larvae_analysis_acid_data_final.ipynb” notebook (see comments). 

The “motifs_compare” folder contains the dictionaries obtained by running BASS on the exploratory and aversive datasets (five trials each). 
