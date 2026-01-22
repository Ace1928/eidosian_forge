import numpy as np
from ._ckdtree import cKDTree, cKDTreeNode
def sparse_distance_matrix(self, other, max_distance, p=2.0, output_type='dok_matrix'):
    """Compute a sparse distance matrix.

        Computes a distance matrix between two KDTrees, leaving as zero
        any distance greater than max_distance.

        Parameters
        ----------
        other : KDTree

        max_distance : positive float

        p : float, 1<=p<=infinity
            Which Minkowski p-norm to use.
            A finite large p may cause a ValueError if overflow can occur.

        output_type : string, optional
            Which container to use for output data. Options: 'dok_matrix',
            'coo_matrix', 'dict', or 'ndarray'. Default: 'dok_matrix'.

            .. versionadded:: 1.6.0

        Returns
        -------
        result : dok_matrix, coo_matrix, dict or ndarray
            Sparse matrix representing the results in "dictionary of keys"
            format. If a dict is returned the keys are (i,j) tuples of indices.
            If output_type is 'ndarray' a record array with fields 'i', 'j',
            and 'v' is returned,

        Examples
        --------
        You can compute a sparse distance matrix between two kd-trees:

        >>> import numpy as np
        >>> from scipy.spatial import KDTree
        >>> rng = np.random.default_rng()
        >>> points1 = rng.random((5, 2))
        >>> points2 = rng.random((5, 2))
        >>> kd_tree1 = KDTree(points1)
        >>> kd_tree2 = KDTree(points2)
        >>> sdm = kd_tree1.sparse_distance_matrix(kd_tree2, 0.3)
        >>> sdm.toarray()
        array([[0.        , 0.        , 0.12295571, 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.28942611, 0.        , 0.        , 0.2333084 , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.24617575, 0.29571802, 0.26836782, 0.        , 0.        ]])

        You can check distances above the `max_distance` are zeros:

        >>> from scipy.spatial import distance_matrix
        >>> distance_matrix(points1, points2)
        array([[0.56906522, 0.39923701, 0.12295571, 0.8658745 , 0.79428925],
           [0.37327919, 0.7225693 , 0.87665969, 0.32580855, 0.75679479],
           [0.28942611, 0.30088013, 0.6395831 , 0.2333084 , 0.33630734],
           [0.31994999, 0.72658602, 0.71124834, 0.55396483, 0.90785663],
           [0.24617575, 0.29571802, 0.26836782, 0.57714465, 0.6473269 ]])

        """
    return super().sparse_distance_matrix(other, max_distance, p, output_type)