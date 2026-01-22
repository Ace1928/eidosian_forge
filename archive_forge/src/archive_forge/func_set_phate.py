import numpy as np
import scipy
import sklearn
import meld
import graphtools as gt
def set_phate(self, data_phate):
    """Short summary.

        Parameters
        ----------
        data_phate : array, shape=[n_samples, 3]
            PHATE embedding for input data.

        Returns
        -------
        data_phate : array, shape=[n_samples, 3]
            Normalized PHATE embedding.

        """
    if not data_phate.shape[1] == 3:
        raise ValueError('data_phate must have 3 dimensions')
    if not np.isclose(data_phate.mean(), 0):
        data_phate = scipy.stats.zscore(data_phate, axis=0)
    self.data_phate = data_phate