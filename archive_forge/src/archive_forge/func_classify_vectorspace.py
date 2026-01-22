from nltk.cluster.util import Dendrogram, VectorSpaceClusterer, cosine_distance
def classify_vectorspace(self, vector):
    best = None
    for i in range(self._num_clusters):
        centroid = self._centroids[i]
        dist = cosine_distance(vector, centroid)
        if not best or dist < best[0]:
            best = (dist, i)
    return best[1]