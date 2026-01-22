import numpy as np
import pandas as pd
from scipy.special import inv_boxcox
from scipy.stats import (
from scipy.stats.distributions import rv_frozen
from statsmodels.base.data import PandasData
from statsmodels.base.model import Results
from statsmodels.base.wrapper import (
@property
def params_formatted(self):
    """
        DataFrame containing all parameters

        Contains short names and a flag indicating whether the parameter's
        value was optimized to fit the data.
        """
    return self._params_formatted