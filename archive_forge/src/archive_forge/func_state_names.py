import os
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from statsmodels import datasets
from statsmodels.tools import add_constant
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.statespace import mlemodel
@property
def state_names(self):
    state_names = []
    for i in range(self.k_endog):
        endog_name = self.endog_names[i]
        state_names += ['intercept.%s' % endog_name]
        state_names += ['L1.{}->{}'.format(other_name, endog_name) for other_name in self.endog_names]
    return state_names