import numpy as np 
# Scipy lirbaries just use to calculate the distance between 2 points
from scipy.spatial.distance import cdist

def kmeans_init_centers(X, k):
    return X[np.random.choice(X.shape[0], k, replace=False)]

def kmeans_assign_labels(X, centers):
    D = cdist(X, centers)
    return np.argmin(D, axis = 1)

def kmeans_update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        Xk = X[labels == k, :]
        centers[k,:] = np.mean(Xk, axis = 0)
    return centers

def has_converged(centers, new_centers):
    return (set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers]))

def kmeans(X, K):
    centers = [kmeans_init_centers(X, K)]
    labels = []
    it = 0 
    while True:
        labels.append(kmeans_assign_labels(X, centers[-1]))
        new_centers = kmeans_update_centers(X, labels[-1], K)
        if has_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
        it += 1
    return (centers, labels, it)

if __name__ == '__main__':
    X = [[255, 170, 12], 
        [23, 25, 35],
        [250, 220, 13],
        [34, 34, 56],
        [63, 77, 17]]
    X = np.array(X)
    (centers, labels, it) = kmeans(X, 5)
    print(centers)
    print(labels)
    print(it)
