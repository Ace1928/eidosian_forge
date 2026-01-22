import numpy as np
import pandas as pd
from scipy.special import inv_boxcox
from scipy.stats import (
from scipy.stats.distributions import rv_frozen
from statsmodels.base.data import PandasData
from statsmodels.base.model import Results
from statsmodels.base.wrapper import (
@property
def fcastvalues(self):
    """
        An array of the forecast values
        """
    return self._fcastvalues