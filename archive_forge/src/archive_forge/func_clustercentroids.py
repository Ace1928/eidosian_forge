import numbers
from . import _cluster  # type: ignore
def clustercentroids(self, clusterid=None, method='a', transpose=False):
    """Calculate the cluster centroids and return a tuple (cdata, cmask).

        The centroid is defined as either the mean or the median over all
        items for each dimension.

        Keyword arguments:
         - data: nrows x ncolumns array containing the expression data
         - mask: nrows x ncolumns array of integers, showing which data
           are missing. If mask[i, j]==0, then data[i, j] is missing.
         - transpose: if False, gene (row) clusters are considered;
                      if True, sample (column) clusters are considered.
         - clusterid: array containing the cluster number for each gene or
           sample. The cluster number should be non-negative.
         - method: specifies how the centroid is calculated:
           - method == 'a': arithmetic mean over each dimension. (default)
           - method == 'm': median over each dimension.

        Return values:
         - cdata: 2D array containing the cluster centroids. If transpose
           is False, then the dimensions of cdata are nclusters x ncolumns.
           If transpose is True, then the dimensions of cdata are nrows x
           nclusters.
         - cmask: 2D array of integers describing which items in cdata,
           if any, are missing.
        """
    return clustercentroids(self.data, self.mask, clusterid, method, transpose)