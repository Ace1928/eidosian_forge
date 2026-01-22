import re
import importlib as im
import logging
import types
import json
from itertools import combinations
from pyomo.common.dependencies import (
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.environ import Block, ComponentUID
import pyomo.contrib.parmest.utils as utils
import pyomo.contrib.parmest.graphics as graphics
from pyomo.dae import ContinuousSet
def likelihood_ratio_test(self, obj_at_theta, obj_value, alphas, return_thresholds=False):
    """
        Likelihood ratio test to identify theta values within a confidence
        region using the :math:`\\chi^2` distribution

        Parameters
        ----------
        obj_at_theta: pd.DataFrame, columns = theta_names + 'obj'
            Objective values for each theta value (returned by
            objective_at_theta)
        obj_value: int or float
            Objective value from parameter estimation using all data
        alphas: list
            List of alpha values to use in the chi2 test
        return_thresholds: bool, optional
            Return the threshold value for each alpha

        Returns
        -------
        LR: pd.DataFrame
            Objective values for each theta value along with True or False for
            each alpha
        thresholds: pd.Series
            If return_threshold = True, the thresholds are also returned.
        """
    assert isinstance(obj_at_theta, pd.DataFrame)
    assert isinstance(obj_value, (int, float))
    assert isinstance(alphas, list)
    assert isinstance(return_thresholds, bool)
    LR = obj_at_theta.copy()
    S = len(self.callback_data)
    thresholds = {}
    for a in alphas:
        chi2_val = scipy.stats.chi2.ppf(a, 2)
        thresholds[a] = obj_value * (chi2_val / (S - 2) + 1)
        LR[a] = LR['obj'] < thresholds[a]
    thresholds = pd.Series(thresholds)
    if return_thresholds:
        return (LR, thresholds)
    else:
        return LR