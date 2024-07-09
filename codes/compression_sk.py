from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

original = plt.imread('puss.jpg')

plt.imshow(original)
plt.show()

original = original / 255
X = original.reshape(-1, 3)

K = 25                       

kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
idx = kmeans.labels_
centroids = kmeans.cluster_centers_

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