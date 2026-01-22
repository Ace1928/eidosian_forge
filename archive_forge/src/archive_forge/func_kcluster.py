import numbers
from . import _cluster  # type: ignore
def kcluster(self, nclusters=2, transpose=False, npass=1, method='a', dist='e', initialid=None):
    """Apply k-means or k-median clustering.

        This method returns a tuple (clusterid, error, nfound).

        Keyword arguments:
         - nclusters: number of clusters (the 'k' in k-means)
         - transpose: if False, genes (rows) are clustered;
                      if True, samples (columns) are clustered.
         - npass: number of times the k-means clustering algorithm is
           performed, each time with a different (random) initial condition.
         - method: specifies how the center of a cluster is found:
           - method == 'a': arithmetic mean
           - method == 'm': median
         - dist: specifies the distance function to be used:
           - dist == 'e': Euclidean distance
           - dist == 'b': City Block distance
           - dist == 'c': Pearson correlation
           - dist == 'a': absolute value of the correlation
           - dist == 'u': uncentered correlation
           - dist == 'x': absolute uncentered correlation
           - dist == 's': Spearman's rank correlation
           - dist == 'k': Kendall's tau
         - initialid: the initial clustering from which the algorithm should
           start. If initialid is None, the routine carries out npass
           repetitions of the EM algorithm, each time starting from a different
           random initial clustering. If initialid is given, the routine
           carries out the EM algorithm only once, starting from the given
           initial clustering and without randomizing the order in which items
           are assigned to clusters (i.e., using the same order as in the data
           matrix). In that case, the k-means algorithm is fully deterministic.

        Return values:
         - clusterid: array containing the number of the cluster to which each
           gene/sample was assigned in the best k-means clustering
           solution that was found in the npass runs;
         - error: the within-cluster sum of distances for the returned
           k-means clustering solution;
         - nfound: the number of times this solution was found.
        """
    if transpose:
        weight = self.gweight
    else:
        weight = self.eweight
    return kcluster(self.data, nclusters, self.mask, weight, transpose, npass, method, dist, initialid)