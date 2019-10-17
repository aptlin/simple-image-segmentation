from sklearn.metrics import pairwise_distances_argmin
import numpy as np
import config


class Segmentation:
    def __init__(self, n_clusters, random_seed=0, n_init=10):
        self.n_clusters = n_clusters
        self.random_seed = random_seed
        self.n_init = n_init

    def kmeans(self, points):
        """Adapted from https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
        """
        rng = np.random.RandomState(self.random_seed)
        sample = rng.permutation(len(points))[: self.n_clusters]
        centers = points[sample]
        while True:
            labels = pairwise_distances_argmin(points, centers)
            new_centers = np.array(
                [points[labels == i].mean(0) for i in range(self.n_clusters)]
            )
            if np.all(centers == new_centers):
                break
            centers = new_centers

        return centers, labels

    def segment(self, img):
        height = img.shape[0]
        width = img.shape[1]
        channels = 1 if len(img.shape) < 3 else img.shape[2]
        flat_img = img.reshape((height * width, channels))
        for init in range(self.n_init):
            clusters, centers = self.kmeans(flat_img)
        return clusters, centers
