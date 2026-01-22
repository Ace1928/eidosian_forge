import warnings
import numpy as np
from statsmodels.tools.decorators import cache_readonly
from statsmodels.stats.diagnostic_gen import (
from statsmodels.discrete._diagnostics_count import (
@cache_readonly
def probs_predicted(self):
    if self.y_max is not None:
        kwds = {'y_values': np.arange(self.y_max + 1)}
    else:
        kwds = {}
    return self.results.predict(which='prob', **kwds)