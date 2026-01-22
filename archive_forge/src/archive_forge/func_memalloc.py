import numpy as np
from collections import namedtuple
def memalloc(self, sz):
    """
        Allocates memory on the simulated device
        At present, there is no division between simulated
        host memory and simulated device memory.
        """
    return np.ndarray(sz, dtype='u1')