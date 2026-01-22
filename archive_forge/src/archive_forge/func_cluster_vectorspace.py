from nltk.cluster.util import Dendrogram, VectorSpaceClusterer, cosine_distance
def cluster_vectorspace(self, vectors, trace=False):
    N = len(vectors)
    cluster_len = [1] * N
    cluster_count = N
    index_map = numpy.arange(N)
    dims = (N, N)
    dist = numpy.ones(dims, dtype=float) * numpy.inf
    for i in range(N):
        for j in range(i + 1, N):
            dist[i, j] = cosine_distance(vectors[i], vectors[j])
    while cluster_count > max(self._num_clusters, 1):
        i, j = numpy.unravel_index(dist.argmin(), dims)
        if trace:
            print('merging %d and %d' % (i, j))
        self._merge_similarities(dist, cluster_len, i, j)
        dist[:, j] = numpy.inf
        dist[j, :] = numpy.inf
        cluster_len[i] = cluster_len[i] + cluster_len[j]
        self._dendrogram.merge(index_map[i], index_map[j])
        cluster_count -= 1
        index_map[j + 1:] -= 1
        index_map[j] = N
    self.update_clusters(self._num_clusters)