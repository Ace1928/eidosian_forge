import numpy as np
from numpy.testing import assert_almost_equal
import numpy.testing as npt
import statsmodels.tools.eval_measures as em
from statsmodels.stats.moment_helpers import cov2corr, se_cov
from statsmodels.regression.linear_model import OLS
from statsmodels.sandbox.panel.panel_short import ShortPanelGLS, ShortPanelGLS2
from statsmodels.sandbox.panel.random_panel import PanelSample
import statsmodels.sandbox.panel.correlation_structures as cs
import statsmodels.stats.sandwich_covariance as sw
Test for short_panel and panel sandwich

Created on Fri May 18 13:05:47 2012

Author: Josef Perktold

moved example from main of random_panel
