import numpy as np

def svm_is_unknown_from_probs(probs, threshold):
    """
    Detect unknown sample for SVM based on max probability.
    
    Args:
        probs (np.array): 1D array from model.predict_proba
        threshold (float): max probability threshold to consider unknown

    Returns:
        (is_unknown: bool, max_prob: float)
    """
    maxp = float(np.max(probs))
    return (maxp < threshold), maxp

def knn_is_unknown_from_mean_dist(mean_dist, threshold):
    """
    Detect unknown sample for k-NN based on mean neighbor distance.
    
    Args:
        mean_dist (float): mean distance to k nearest neighbors
        threshold (float): distance threshold to consider unknown

    Returns:
        (is_unknown: bool, mean_dist: float)
    """
    return (mean_dist > threshold), float(mean_dist)

def suggest_svm_threshold_from_val(svm_val_maxp_array, target_reject_rate=0.02):
    """
    Suggest SVM threshold using validation max probability array.
    
    Args:
        svm_val_maxp_array (np.array): max probabilities on validation set
        target_reject_rate (float): fraction of samples to reject

    Returns:
        float: threshold
    """
    thr = np.quantile(svm_val_maxp_array, target_reject_rate)
    return float(thr)

def suggest_knn_threshold_from_val(knn_val_mean_dist_array, target_reject_rate=0.02):
    """
    Suggest k-NN threshold using validation mean distances.
    
    Args:
        knn_val_mean_dist_array (np.array): mean distances on validation set
        target_reject_rate (float): fraction of samples to reject

    Returns:
        float: threshold
    """
    thr = np.quantile(knn_val_mean_dist_array, 1.0 - target_reject_rate)
    return float(thr)