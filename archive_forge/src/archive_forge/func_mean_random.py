import numpy as np
import numpy.linalg as L
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.tools.decorators import cache_readonly
def mean_random(self, idx='lastexog'):
    if idx == 'lastexog':
        meanr = self.params[-self.model.k_exog_re:]
    elif isinstance(idx, list):
        if not len(idx) == self.model.k_exog_re:
            raise ValueError('length of idx different from k_exog_re')
        else:
            meanr = self.params[idx]
    else:
        meanr = np.zeros(self.model.k_exog_re)
    return meanr