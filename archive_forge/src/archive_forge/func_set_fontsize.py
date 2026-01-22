from statsmodels.compat.python import lrange
import numpy as np
import statsmodels.tsa.vector_ar.util as util
def set_fontsize(self, size):
    import matplotlib as mpl
    old_size = mpl.rcParams['font.size']
    mpl.rcParams['font.size'] = size

    def revert():
        mpl.rcParams['font.size'] = old_size
    self._inverse_actions.append(revert)