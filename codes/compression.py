from kmeans import *

original = plt.imread('puss.jpg')

plt.imshow(original)
plt.show()

# Reshape the original image
original = original / 255
X = original.reshape(-1, 3)

# Run your K-Means algorithm on this data
# You should try different values of K and max_iters here
K = 20                       
max_iters = 20      

# Using the function you have implemented above. 
initial_centroids = kMeans_init_centroids(X, K) 

# Run K-Means - this takes a couple of minutes
centroids, idx = run_kMeans(X, initial_centroids, max_iters) 

# Compress the image
X_recovered = centroids[idx, :]
X_recovered = np.reshape(X_recovered, original.shape) 

# Display the original and the compressed image
fig, ax = plt.subplots(1,2, figsize=(8,8))
plt.axis('off')

ax[0].imshow(original)
ax[0].set_title('Original')
ax[0].set_axis_off()

# Display compressed image
ax[1].imshow(X_recovered)
ax[1].set_title('Compressed with %d colours'%K)
ax[1].set_axis_off()

plt.show()