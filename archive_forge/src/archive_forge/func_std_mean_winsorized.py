import numbers
import numpy as np
@property
def std_mean_winsorized(self):
    """standard error of winsorized mean
        """
    std_ = np.sqrt(self.var_winsorized / self.nobs)
    std_ *= (self.nobs - 1) / (self.nobs_reduced - 1)
    return std_