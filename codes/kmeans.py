import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine

def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example
    
    Args:
        X (ndarray): (m, n) Input values      
        centroids (ndarray): k centroids
    
    Returns:
        idx (array_like): (m,) closest centroids
    """
    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)

    for i in range(X.shape[0]):
        # Array to hold distance between X[i] and each centroids[j]
        distance = [] 
        for j in range(K):
            norm_ij = np.linalg.norm(X[i] - centroids[j])
            distance.append(norm_ij)
        idx[i] = np.argmin(distance)   

    return idx

def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the 
    data points assigned to each centroid.
    
    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each 
                       example in X. Concretely, idx[i] contains the index of 
                       the centroid closest to example i
        K (int):       number of centroids
    
    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """
    # Useful variables
    m, n = X.shape
    
    # You need to return the following variables correctly
    centroids = np.zeros((K, n))
    
    for k in range(K):   
          points = X[idx == k, :]
          centroids[k] = np.mean(points, axis = 0)

    return centroids

def run_kMeans(X, initial_centroids, max_iters=10):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example

    Args:
        X (ndarray): (m, n) Data points
        initial_centroids (ndarray): (K, n) Initial centroids
        max_iters (int): number of iterations to run
        plot_progress (bool): True to plot progress, False otherwise

    Returns:
        centroids (ndarray): (K, n) Final centroids
        idx (ndarray): (m,) Index of the closest centroid for each example
    """
    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids   
    idx = np.zeros(m)
    
    # Run K-Means
    for i in range(max_iters):
        
        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)
            
        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)

    return centroids, idx

def kMeans_init_centroids(X, K):
    """
    This function initializes K centroids that are to be 
    used in K-Means on the dataset X
    
    Args:
        X (ndarray): Data points 
        K (int):     number of centroids/clusters
    
    Returns:
        centroids (ndarray): Initialized centroids
    """
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    
    # Take the first K examples as centroids
    centroids = X[randidx[:K]]
    
    return centroids

def visualize_kMeans(X, centroids, idx):
    """
    Plots the data points with colors assigned to each centroid.
    
    Args:
        X (ndarray): (m, n) Data points
        centroids (ndarray): (K, n) Centroids
        idx (ndarray): (m,) Index of the closest centroid for each example
    """
    # Useful variables
    K = centroids.shape[0]
    m, n = X.shape

    # pca for dimensionality reduction
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    centroids = pca.transform(centroids)
    
    # Plot the data
    plt.scatter(X[:, 0], X[:, 1], c=idx, cmap='viridis', marker='o')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100)
    plt.show()

def run_example():
    # Load an example dataset
    data = load_wine()
    X = data.data
    y = data.target
    
    # Settings for running K-Means
    K = 3
    max_iters = 10
    
    # For consistency, here we set centroids to specific values
    initial_centroids = kMeans_init_centroids(X, K)
    
    # Run K-Means algorithm
    centroids, idx = run_kMeans(X, initial_centroids, max_iters)

    # Visualize the K-Means result
    visualize_kMeans(X, centroids, idx)


if __name__ == '__main__':
    run_example()
