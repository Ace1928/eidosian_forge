from . import utils
from scipy import sparse
from sklearn import decomposition
from sklearn import random_projection
import numpy as np
import pandas as pd
import sklearn.base
import warnings
@property
def components_(self):
    """Principal axes in feature space, representing directions of maximum variance.

        The components are sorted by explained variance.
        """
    return self.proj_op.inverse_transform(self.pca_op.components_)