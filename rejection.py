import numpy as np

def svm_is_unknown_from_probs(probs, threshold):
  
    maxp = float(np.max(probs))
    return (maxp < threshold), maxp

def knn_is_unknown_from_mean_dist(mean_dist, threshold):
   
    return (mean_dist > threshold), float(mean_dist)

def suggest_svm_threshold_from_val(svm_val_maxp_array, target_reject_rate=0.02):
 
    thr = np.quantile(svm_val_maxp_array, target_reject_rate)
    return float(thr)

def suggest_knn_threshold_from_val(knn_val_mean_dist_array, target_reject_rate=0.02):
    
    thr = np.quantile(knn_val_mean_dist_array, 1.0 - target_reject_rate)
    return float(thr)