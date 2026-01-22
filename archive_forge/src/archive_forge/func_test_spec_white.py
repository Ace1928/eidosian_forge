import json
import os
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from statsmodels.datasets import macrodata, sunspots
from statsmodels.regression.linear_model import OLS
import statsmodels.stats.diagnostic as smsdia
import statsmodels.stats.outliers_influence as oi
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.tools import Bunch, add_constant
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
def test_spec_white():
    resdir = os.path.join(cur_dir, 'results')
    wsfiles = ['wspec1.csv', 'wspec2.csv', 'wspec3.csv', 'wspec4.csv']
    for file in wsfiles:
        mdlfile = os.path.join(resdir, file)
        mdl = np.asarray(pd.read_csv(mdlfile))
        lastcol = mdl.shape[1] - 1
        dv = mdl[:, lastcol]
        design = np.concatenate((np.ones((mdl.shape[0], 1)), np.delete(mdl, lastcol, 1)), axis=1)
        resids = dv - np.dot(design, np.linalg.lstsq(design, dv, rcond=-1)[0])
        wsres = smsdia.spec_white(resids, design)
        if file == 'wspec1.csv':
            assert_almost_equal(wsres, [3.251, 0.661, 5], decimal=3)
        elif file == 'wspec2.csv':
            assert_almost_equal(wsres, [6.07, 0.733, 9], decimal=3)
        elif file == 'wspec3.csv':
            assert_almost_equal(wsres, [6.767, 0.454, 7], decimal=3)
        else:
            assert_almost_equal(wsres, [8.462, 0.671, 11], decimal=3)