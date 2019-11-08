#Code for the paper Margin Maximization as Lossless Maximal Compression
#by N. Nikolaou, H. Reeve & G. Brown,
#Submitted in MLJ, S.I. ECML-PKDD 2020 on Nov 8 2019. 

from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.metrics import zero_one_loss
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio

n_runs = 100 #number of runs; each run uses a different train/test split
n_est = 100 #(maximum) number of boosting rounds

bin_count = 100 #every feature will be discretized to bin_count equal-sized bins
score_F_bins = 100 # scores (ensemble model outputs) will be discretized to score_F_bins equal-sized bins

#Gradient boosting hyperparameters: loss function, number of boosting rounds, maximum tree depth, learning rate (1 for no shrinkage, loss function: for AdaBoost use 'exponential'):
params = {'loss': 'exponential', 'n_estimators': n_est, 'max_depth': 6 , 'learning_rate' : 1, 'subsample' : 1} 

###############################################################################

#Load one (or more...in which case finish loop below) of these datasets:
#dataset_list = ["survival", "ionosphere", "congress_orig", "liver", "pima",
#                "parkinsons", "landsatM", "krvskp", "heart_orig", "wdbc_orig",
#                "german_credit", "sonar_orig", "semeion_orig", "spliceM",
#                "waveformM", "spambase", "mushroom", "musk2"]
dataset_name = ["heart_orig"] 
#survival, congress_orig, landsatM --discrete; no further discretization needed unless to bin to fewer than 10 values
#survival, pima, liver spambase --noisy for bin_count < 100; not used in experiments shown in paper (we cover noiseless case); will be used in future work, included here nonetheless

##for dat_num in range(len(dataset_name)):
##  data_path = os.path.join(os.getcwd(), 'Datasets', dataset_list[dat_num]+'.mat')
##  ...

data_path = os.path.join(os.getcwd(), 'Datasets', dataset_name[0]+'.mat')
mat_contents = sio.loadmat(data_path)
X_undisc = mat_contents['data']
y = np.asarray([float(i) for i in mat_contents['labels'].ravel()]) #loadmat loads the labels as a list of uints which we turn into an array of floats
y[y!=1] = 0 #binarize multiclass datasets using 0/1 encoding

###############################################################################

#OR make an artificial dataset using scikit-learn's make_classification():
#from sklearn.datasets import make_classification
#X_undisc, y = make_classification(n_samples=2000, n_features=20, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=2,  weights=None, flip_y=0.01)#weights=[0.1,0.9]

###############################################################################

#Discretize features:
num_features = X_undisc.shape[1] 
   
X = np.zeros(X_undisc.shape)
for i in range(num_features):
    if np.max(np.absolute(X_undisc[:,i]))==0:#to avoid division by 0 when entire column is 0
       X[:,i] = X_undisc[:,i]
    else: 
        X[:,i] = X_undisc[:,i] / np.max(np.absolute(X_undisc[:,i]))
    X_i_binned = np.histogram(X_undisc[:,i], bins=bin_count)
    X[:,i]  = np.fmin(np.digitize(X_undisc[:,i], X_i_binned[1]), bin_count)

#Variables that will store quantities of interest:    
margin = np.zeros((n_runs,n_est)) #average margin of model on training set
margin_hist = np.zeros((n_runs,n_est,int(X_undisc.shape[0]*0.5))) #margins of model on all training examples
MI_FY = np.zeros((n_runs,n_est)) #I(F;Y), i.e. mutual information between score produced by model on training dataset (score_F_i_disc) & target label of training examples (y_train)
MI_XF = np.zeros((n_runs,n_est)) #I(X;F), i.e. mutual information between score produced by model on training dataset (score_F_i_disc) & (joint) feature vector of training examples (X_train)
error_tr = np.zeros((n_runs,n_est)) #training error of model
error_te = np.zeros((n_runs,n_est)) #test error of model
entropy_F = np.zeros((n_runs,n_est)) #H(F), i.e. entropy of scores produced by model on training dataset (score_F_i_disc)
MI_XY = np.zeros(n_runs)  #I(X;Y), , i.e. mutual information between (joint) feature vector of training examples (X_train) & target label of training examples (y_train)
entropy_Y = np.zeros(n_runs) #H(Y), i.e. entropy of target label of training examples (y_train)
entropy_X = np.zeros(n_runs) #H(X), i.e. entropy of (joint) feature vectors in dataset (X_train)

max_score = 0 # will store maximal score assigned by model
min_score = 0 # will store minimal score assigned by model

#On each run perform a train/test split, train ensemble model and store all quantities of interest:
for run in range(n_runs):
    
    #Perform train/test split:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    y_train_sign = 2*y_train-1  #switch to -1/1 encoding
    
    #Form a joint r.v. out of all the features (all features have been discretized to bin_count values each):
    X_train_joint  = np.zeros(X_train[:,0].shape)
    for i in range(num_features): 
        X_train_joint  = X_train_joint + (X_train[:,i]-1)*((bin_count)**i)

    #Build ensemble
    clf = ensemble.GradientBoostingClassifier(**params)
    clf.fit(X_train, y_train)

    #Calculate scores and predictions for each boosting round (i.e. 'staged'):
    score_F_bin_list = np.linspace(0, 1.0, num=score_F_bins+1)[1:]

    #for i, score_F_i in enumerate(clf.staged_predict_proba(X_train)): #A bad built-in for calculating probability estimates (i.e. in [0,1])
    for i, score_F_i in enumerate(clf.staged_decision_function(X_train)): #A better built-in for calculating raw (i.e. real-valued) scores (Actual F(x) of Boosting divided by sum of alphas -- basicaly equals normalized_margin(x)/y )
       
        #Discretize real-valued scores for estimating information-theoretic quantities:
        score_F_i_binned = np.histogram(score_F_i[:,0], bins=score_F_bin_list)  
        score_F_i_disc = np.fmin(np.digitize(score_F_i[:,0], score_F_i_binned[1]), len(score_F_bin_list))
        
        #Estimate model-dependent information-theoretic quantities (both model and training dataset vary per run): 
        MI_FY[run,i] = mutual_info_score(score_F_i_disc, y_train)
        MI_XF[run,i] = mutual_info_score(score_F_i_disc, X_train_joint)
        entropy_F[run,i] = mutual_info_score(score_F_i_disc, score_F_i_disc) #I(F;F) = H(F)
        #To get normalized margins must normalize scores (can also use raw, i.e. real-valued scores):
        max_score = max(max_score, np.max(score_F_i_disc)) #largest score ever assigned by model
        min_score = min(min_score, np.min(score_F_i_disc)) #lowest score ever assigned by model
        score_range = max_score - min_score
        score_F_i_norm = 2*((score_F_i_disc - min_score)/score_range) - 1 #normalize by range within [-1,1]
        
        #Compute model's margins on dataset (both model and training dataset vary per run):        
        margin[run,i] = np.nanmean(np.divide((score_F_i_norm).T, y_train_sign))
        margin_hist[run,i,:] = np.divide((score_F_i_norm).T, y_train_sign)
        
    
    #Estimate model-independent information-theoretic quantities (training dataset varies per run):     
    MI_XY[run] = mutual_info_score(X_train_joint, y_train)
    entropy_Y[run] = mutual_info_score(y_train, y_train) #I(Y;Y) = H(Y)
    entropy_X[run] = mutual_info_score(X_train_joint, X_train_joint) #I(X;X) = H(X)
     
    #Compute model's predictions on dataset (both model and training dataset vary per run):
    for i, pred_y_train in enumerate(clf.staged_predict(X_train)):
        error_tr[run,i] = zero_one_loss(pred_y_train, y_train)
    for i, pred_y_test in enumerate(clf.staged_predict(X_test)): 
        error_te[run,i] = zero_one_loss(pred_y_test, y_test)
     
#############################
        
#Compute mean and standard deviation of quantities of interest: 
margin_mean = np.nanmean(margin, axis=0)
margin_hist_mean = np.nanmean(margin_hist, axis=0)
MI_XF_mean = np.nanmean(MI_XF, axis=0)
MI_FY_mean = np.nanmean(MI_FY, axis=0)
avg_pairwise_distance_MI = (np.abs(MI_XF_mean-MI_FY_mean) + np.abs(MI_XF_mean-np.tile(np.nanmean(entropy_Y),(n_est,))) + np.abs(MI_FY_mean-np.tile(np.nanmean(entropy_Y),(n_est,))))/3 #average pairwise distance among MI_XF_mean, MI_FY_mean and MI_XY (if zero, then LMC)
entropy_F_mean = np.nanmean(entropy_F, axis=0)
error_tr_mean = np.nanmean(error_tr, axis=0)
error_te_mean = np.nanmean(error_te, axis=0)

margin_std = np.nanstd(margin, axis=0)
margin_hist_std = np.nanstd(margin_hist, axis=0)
MI_XF_std = np.nanstd(MI_XF, axis=0)
MI_FY_std = np.nanstd(MI_FY, axis=0)
#std_pairwise_distance_MI = ... # not used; define if needed
entropy_F_std = np.nanstd(entropy_F, axis=0)
error_tr_std = np.nanstd(error_tr, axis=0)
error_te_std = np.nanstd(error_te, axis=0)

#Find index (round of boosting) at which each quantity of interest --on average-- reaches its minimum/maximum:
index_MI_XF_mean_max = np.where(MI_XF_mean==np.max(MI_XF_mean, axis=0))[0][0]
index_MI_XF_mean_min = np.where(MI_XF_mean==np.min(MI_XF_mean, axis=0))[0][0]
index_MI_FY_mean_max = np.where(MI_FY_mean==np.max(MI_FY_mean, axis=0))[0][0]
index_error_tr_first_min = np.where(error_tr_mean==np.min(error_tr_mean))[0][0]
index_error_te_first_min = np.where(error_te_mean==np.min(error_te_mean))[0][0]
index_entropy_F_first_max = np.where(entropy_F_mean==np.max(entropy_F_mean))[0][0]
#Round reaching LMC point (index at which MI_FY_mean is first maximized while MI_XF_mean being also minimized):
try:
    index_LMC = np.intersect1d(np.where(MI_FY_mean==np.max(MI_FY_mean, axis=0))[0], np.where(MI_XF_mean==np.min(MI_XF_mean, axis=0))[0])[0]
except IndexError as e: #If no such point found, find point closest to it:
    index_LMC = np.where(avg_pairwise_distance_MI==np.min(avg_pairwise_distance_MI))[0][0] #Round reaching point in which (sum / average of) pairwise distances of MI_XF_mean, MI_FY_mean and MI_XY is minimal (closest point to LMC)
index_margin_mean_max = np.where(margin_mean==np.max(margin_mean, axis=0))[0][0] #remove [0] ...

#############################

#Plot results:

##Code below produces continuous shaded equivalents of errorbars (not used in paper; error plots not included):
#def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None, linestyle = None):
#    ax = ax if ax is not None else plt.gca()
#    if color is None:
#        color = next(ax._get_lines.prop_cycler)['color']
#    if np.isscalar(yerr) or len(yerr) == len(y):
#        ymin = y - yerr
#        ymax = y + yerr
#    elif len(yerr) == 2:
#        ymin, ymax = yerr
#    ax.plot(x, y, color=color, linestyle=linestyle)
#    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)
#
##Plot the evolution of the ensemble's training and test error and IP quantities (not included in paper):
#plt.figure(1)
#errorfill(np.arange(1, n_est+1, 1), error_tr_mean, error_tr_std, color='blue') #show training error (mean +/- standard deviation)   #tr_error_plt, = plt.plot(np.arange(1, n_est+1, 1), error_tr_mean, 'b', linewidth=3) #show training error (mean)
#errorfill(np.arange(1, n_est+1, 1), error_te_mean, error_te_std, color='red') #show test error (mean +/- standard deviation)   #te_error_plt, =plt.plot(np.arange(1, n_est+1, 1), error_te_mean, 'r', linewidth=3) #show test error (mean)
#plt.axvline(x=index_error_tr_first_min+1, color = 'k', linestyle='--', linewidth=2) #show round of minimal training error
#plt.axvline(x=index_error_te_first_min+1, color = 'm', linestyle='--', linewidth=2) #show round of minimal test error
#plt.axvline(x=index_margin_mean_max[0]+1, color = 'g', linestyle=':', linewidth=2) #show round of margin maximization
#plt.axvline(x=index_LMC+1, color = 'gray', linestyle='-', linewidth=2) #show round of LMC
##plt.axhline(y=np.min(MI_XF_mean, axis=0)/np.nanmean(entropy_X, axis=0), color = 'k') #show minimal value of I_S(F;X) / H_S(X) (maximal compression)
##plt.axvline(x=index_MI_XF_mean_max+1, color = 'y', linestyle='--', linewidth=2) #show round of minimally compressed model
#plt.legend(labels= ["Training Error", "Test Error", "Min. Training Error Round", "Min. Test Error Round", "Max. Margin Round", "LMC Round"], loc=0)
#plt.gca().set_xlim(left=1)
#plt.title("Evolution of the ensemble's average training & test error")            
#plt.ylabel("Classification Error")
#plt.xlabel("Boosting Round")
#plt.show()

#Plot the trajectory of the model in the entropy-normalized information plane, encoding round of boosting in color of point:
plt.figure(2)
x = MI_XF_mean/np.nanmean(entropy_X, axis=0)
y = MI_FY_mean/np.nanmean(entropy_Y)
plt.xlim(0.98*np.nanmean(MI_XY)/np.nanmean(entropy_X, axis=0), 1.02*x.max()) 
plt.ylim(0.98*y.min(), 1.02*y.max())
plt.plot(x,y)
plt.plot(np.nanmean(MI_XY)/np.nanmean(entropy_X, axis=0), 1, marker='*', markersize=12, markeredgecolor='gray', color="red")  #plot LMC point for noiseless datasets
plt.plot(MI_XF_mean[index_error_tr_first_min]/np.nanmean(entropy_X, axis=0), MI_FY_mean[index_error_tr_first_min]/np.nanmean(entropy_Y), marker='o', markersize=6, color="black")  #plot point of training error minimization (point of achieving losslessness)
plt.plot(MI_XF_mean[index_error_te_first_min]/np.nanmean(entropy_X, axis=0), MI_FY_mean[index_error_te_first_min]/np.nanmean(entropy_Y), marker='s', markerfacecolor='none', markersize=14, color="magenta") #plot point of test error minimization (point of maximal generalization)
#plt.plot(MI_XF_mean[index_MI_XF_mean_max]/np.nanmean(entropy_X, axis=0), MI_FY_mean[index_MI_XF_mean_max]/np.nanmean(entropy_Y), marker='p', markersize=8, color="yellow") #plot point of minimally compressed model
#plt.plot(MI_XF_mean[index_entropy_F_first_max]/np.nanmean(entropy_X, axis=0), MI_FY_mean[index_entropy_F_first_max]/np.nanmean(entropy_Y), marker='o', markersize=8, color="yellow")#plot point with maximal entropy of F (should be same as above...)
#plt.plot(np.nanmean(MI_XY)/np.nanmean(entropy_X, axis=0), np.nanmean(MI_XY, axis=0)/np.nanmean(entropy_Y, axis=0), marker='d', markersize=12, color="red") #plot LMC point for general dataset (coincides with the 1st point s.t. rounding)
#plt.plot(MI_XF_mean[index_LMC]/np.nanmean(entropy_X, axis=0), MI_FY_mean[index_LMC]/np.nanmean(entropy_Y), marker='o', markersize=8, color="red")#plot point in trajectory that is closest to LMC point (coincides with the above s.t. rounding)
plt.plot(MI_XF_mean[index_margin_mean_max]/np.nanmean(entropy_X, axis=0), MI_FY_mean[index_margin_mean_max]/np.nanmean(entropy_Y), marker='o', markersize=20, fillstyle = 'none', color="green")#plot point of average margin maximization
plt.title("Average trajectory of the ensemble in the normalized information plane")            
plt.ylabel("$I_S(F;Y) / H_S(Y)$")
plt.xlabel("$I_S(F;X) / H_S(X)$")
plt.show()    

##Plot the evolution of the ensemble's average margin and IP quantities (not included in paper):
#plt.figure(3)
#errorfill(np.arange(1, n_est+1, 1), margin_mean, margin_std, color='blue') #show average training margin per round (mean +/- standard deviation) #margin_mean_plt, = plt.plot(np.arange(1, n_est+1, 1), margin_mean, 'b', linewidth=3) #show average training margin per round (mean)
#plt.axvline(x=index_error_tr_first_min+1, color = 'k', linestyle='--', linewidth=2) #show round of minimal training error
#plt.axvline(x=index_error_te_first_min+1, color = 'm', linestyle='--', linewidth=2) #show round of minimal test error
#plt.axvline(x=index_margin_mean_max[0]+1, color = 'green', linestyle='-', linewidth=2) #show round of margin maximization
#plt.axvline(x=index_LMC+1, color = 'red', linestyle=':', linewidth=2) #show round of LMCplt.legend(labels= ["Average Margin", "$I_S(F;Y) / H_S(Y)$", "$I_S(F;X) / H_S(X)$", "Max. Training Accuracy Round", "Max. Test Accuracy Round", "Max. Compression"], loc=0)
#plt.gca().set_xlim(left=1)
##plt.set_yscale('log')
#plt.title("Evolution of the ensemble's average margin and normalized $I_S(F;Y)$ & $I_S(F;X)$")            
##plt.ylabel("Average margin")
#plt.xlabel("Boosting Round")
#plt.show()

#Plot some random trajectories (first k):, but also show position in IP of model attaining minimal test & training error as well as maximal margin per individual run
k = 6
fig2 = plt.figure()
fig2.subplots_adjust(hspace=0.3, wspace=0.3)
fig2.suptitle("Some individual trajectories of the ensemble in the normalized information plane", fontsize=12)
for i in range(k):
    ax = fig2.add_subplot(2, 3, i+1)
    x = MI_XF[i,:]/entropy_X[i]
    y = MI_FY[i,:]/entropy_Y[i]
    plt.xlim(0.9*x.min(), 1.1*x.max()) 
    plt.ylim(0.9*y.min(), 1.1*y.max())
    plt.plot(x,y)
    plt.plot(MI_XY[i]/entropy_X[i], 1, marker='*', markersize=12, markeredgecolor='gray', color="red") #plot LMC point for noiseless datasets
    
    #find indices of points of interest per run:
    index_error_tr_first_min_trajectory = np.where(error_tr[i,:]==np.min(error_tr[i,:]))[0][0]
    index_error_te_first_min_trajectory = np.where(error_te[i,:]==np.min(error_te[i,:]))[0][0]
    index_margin_mean_max_trajectory = np.where(margin[i,:]==np.max(margin[i,:]))[0][0] #for Landsat, semeion remove [0] here...
    
    plt.plot(MI_XF[i,index_error_tr_first_min_trajectory]/entropy_X[i], MI_FY[i, index_error_tr_first_min_trajectory]/entropy_Y[i], marker='o', markersize=6, color="black")  #plot point of training error minimization (point of achieving losslessness)
    plt.plot(MI_XF[i,index_error_te_first_min_trajectory]/entropy_X[i], MI_FY[i, index_error_te_first_min_trajectory]/entropy_Y[i], marker='s', markerfacecolor='none', markersize=14, color="magenta") #plot point of test error minimization (point of maximal generalization)
    plt.plot(MI_XF[i,index_margin_mean_max_trajectory]/entropy_X[i], MI_FY[i, index_margin_mean_max_trajectory]/entropy_Y[i], marker='o', markersize=20, fillstyle = 'none', color="green") #plot point of average margin maximization
    plt.title("Run "+str(i+1), fontsize=10)            
    plt.ylabel("$I_S(F;Y) / H_S(Y)$")
    plt.xlabel("$I_S(F;X) / H_S(X)$")
plt.show() 

