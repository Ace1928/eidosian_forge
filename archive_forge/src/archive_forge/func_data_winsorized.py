import numbers
import numpy as np
@property
def data_winsorized(self):
    """winsorized data
        """
    lb = np.expand_dims(self.lowerbound, self.axis)
    ub = np.expand_dims(self.upperbound, self.axis)
    return np.clip(self.data_sorted, lb, ub)