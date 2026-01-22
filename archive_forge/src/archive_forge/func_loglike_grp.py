import numpy as np
import statsmodels.base.model as base
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.discrete.discrete_model import (MultinomialResults,
import collections
import warnings
import itertools
def loglike_grp(self, grp, params):
    ofs = None
    if hasattr(self, 'offset'):
        ofs = self._offset_grp[grp]
    llg = np.dot(self._xy[grp], params)
    if ofs is not None:
        llg += self._endofs[grp]
    llg -= np.log(self._denom(grp, params, ofs))
    return llg