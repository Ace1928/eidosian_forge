import numpy as np
import statsmodels.base.model as base
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.discrete.discrete_model import (MultinomialResults,
import collections
import warnings
import itertools
def score_grp(self, grp, params):
    ofs = 0
    if hasattr(self, 'offset'):
        ofs = self._offset_grp[grp]
    d, h = self._denom_grad(grp, params, ofs)
    return self._xy[grp] - h / d