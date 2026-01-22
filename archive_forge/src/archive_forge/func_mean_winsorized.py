import numbers
import numpy as np
@property
def mean_winsorized(self):
    """mean of winsorized data
        """
    return np.mean(self.data_winsorized, self.axis)