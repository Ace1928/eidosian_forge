from collections import Counter
import numpy as np
from scipy import sparse
from pygsp import utils
from . import fourier, difference  # prevent circular import in Python < 3.5
@property
def d(self):
    """The degree (the number of neighbors) of each node."""
    if not hasattr(self, '_d'):
        self._d = np.asarray(self.A.sum(axis=1)).squeeze()
    return self._d