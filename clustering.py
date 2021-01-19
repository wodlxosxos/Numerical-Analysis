import numpy as np
import cv2
from skimage import segmentation, color, io
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans

def k_mean_cluster():
    num_cluster = 23
    img = io.imread("./cluster/clusterimage3.jpg")
    img = color.rgb2lab(img)

    originShape = img.shape
    
    flatImg=np.reshape(img, [-1, 3])
    
    km = KMeans(n_clusters = num_cluster)
    
    km.fit(flatImg)
    labels = km.labels_

    cluster_centers = km.cluster_centers_

    segmentedImg = cluster_centers[np.reshape(labels, originShape[:2])]
    result = color.lab2rgb(segmentedImg)
    plt.imshow(result);plt.show()

def mean_shift():
    #Loading original image
    img = io.imread("./cluster/clusterimage3.jpg")
    img = color.rgb2lab(img)
    
    originShape = img.shape
  
    flatImg=np.reshape(img, [-1, 3])

    bandwidth = estimate_bandwidth(flatImg, quantile=0.04, n_samples=100)    
    ms = MeanShift(bandwidth = bandwidth, bin_seeding=True)

    ms.fit(flatImg)

    labels=ms.labels_

    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)    
    n_clusters_ = len(labels_unique)    
    print("number of estimated clusters : ",  n_clusters_)    

    # Displaying segmented image    
    segmentedImg = cluster_centers[np.reshape(labels, originShape[:2])]
    result = color.lab2rgb(segmentedImg)
    plt.imshow(result);plt.show()
    #cv2.imshow('Image',segmentedImg.astype(np.uint8))
    #cv2.waitKey(0)
if __name__== "__main__":
    k_mean_cluster()
    #mean_shift()
