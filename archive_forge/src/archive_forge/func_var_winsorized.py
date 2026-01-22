import numbers
import numpy as np
@property
def var_winsorized(self):
    """variance of winsorized data
        """
    return np.var(self.data_winsorized, ddof=1, axis=self.axis)