from functools import partial
import numpy as np
from . import _catboost
def is_max_optimal(self):
    """
        Returns
        ----------
        bool : True if metric is maximizable, False otherwise
        """
    return _catboost.is_maximizable_metric(str(self))