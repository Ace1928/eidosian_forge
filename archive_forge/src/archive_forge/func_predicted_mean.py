import numpy as np
import pandas as pd
from scipy import stats
@property
def predicted_mean(self):
    """The predicted mean"""
    return self._wrap_pandas(self._predicted_mean, 'predicted_mean')