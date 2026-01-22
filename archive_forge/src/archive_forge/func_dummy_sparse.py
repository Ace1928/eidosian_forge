from statsmodels.compat.python import lrange, lzip
import numpy as np
import pandas as pd
import statsmodels.tools.data as data_util
from pandas import Index, MultiIndex
def dummy_sparse(self, level=0):
    """create a sparse indicator from a group array with integer labels

        Parameters
        ----------
        groups : ndarray, int, 1d (nobs,)
            An array of group indicators for each observation. Group levels
            are assumed to be defined as consecutive integers, i.e.
            range(n_groups) where n_groups is the number of group levels.
            A group level with no observations for it will still produce a
            column of zeros.

        Returns
        -------
        indi : ndarray, int8, 2d (nobs, n_groups)
            an indicator array with one row per observation, that has 1 in the
            column of the group level for that observation

        Examples
        --------

        >>> g = np.array([0, 0, 2, 1, 1, 2, 0])
        >>> indi = dummy_sparse(g)
        >>> indi
        <7x3 sparse matrix of type '<type 'numpy.int8'>'
            with 7 stored elements in Compressed Sparse Row format>
        >>> indi.todense()
        matrix([[1, 0, 0],
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0]], dtype=int8)


        current behavior with missing groups
        >>> g = np.array([0, 0, 2, 0, 2, 0])
        >>> indi = dummy_sparse(g)
        >>> indi.todense()
        matrix([[1, 0, 0],
                [1, 0, 0],
                [0, 0, 1],
                [1, 0, 0],
                [0, 0, 1],
                [1, 0, 0]], dtype=int8)
        """
    indi = dummy_sparse(self.labels[level])
    self._dummies = indi