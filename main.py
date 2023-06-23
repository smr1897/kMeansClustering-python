import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

image = mpl.image.imread('dog.jpg')
# plt.imshow(image)
# print(image.shape)

x = image.reshape(-1,3)
# print(x.shape)

kmeans = KMeans(n_clusters = 2,n_init=10)
kmeans.fit(x) 

segmented_image = kmeans.cluster_centers_[kmeans.labels_]
segmented_image = segmented_image.reshape(image.shape)

plt.imshow(segmented_image/255)
