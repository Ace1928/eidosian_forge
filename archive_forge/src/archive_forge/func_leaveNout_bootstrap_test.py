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
def leaveNout_bootstrap_test(self, lNo, lNo_samples, bootstrap_samples, distribution, alphas, seed=None):
    """
        Leave-N-out bootstrap test to compare theta values where N data points are
        left out to a bootstrap analysis using the remaining data,
        results indicate if theta is within a confidence region
        determined by the bootstrap analysis

        Parameters
        ----------
        lNo: int
            Number of data points to leave out for parameter estimation
        lNo_samples: int
            Leave-N-out sample size. If lNo_samples=None, the maximum number
            of combinations will be used
        bootstrap_samples: int:
            Bootstrap sample size
        distribution: string
            Statistical distribution used to define a confidence region,
            options = 'MVN' for multivariate_normal, 'KDE' for gaussian_kde,
            and 'Rect' for rectangular.
        alphas: list
            List of alpha values used to determine if theta values are inside
            or outside the region.
        seed: int or None, optional
            Random seed

        Returns
        ----------
        List of tuples with one entry per lNo_sample:

        * The first item in each tuple is the list of N samples that are left
          out.
        * The second item in each tuple is a DataFrame of theta estimated using
          the N samples.
        * The third item in each tuple is a DataFrame containing results from
          the bootstrap analysis using the remaining samples.

        For each DataFrame a column is added for each value of alpha which
        indicates if the theta estimate is in (True) or out (False) of the
        alpha region for a given distribution (based on the bootstrap results)
        """
    assert isinstance(lNo, int)
    assert isinstance(lNo_samples, (type(None), int))
    assert isinstance(bootstrap_samples, int)
    assert distribution in ['Rect', 'MVN', 'KDE']
    assert isinstance(alphas, list)
    assert isinstance(seed, (type(None), int))
    if seed is not None:
        np.random.seed(seed)
    data = self.callback_data.copy()
    global_list = self._get_sample_list(lNo, lNo_samples, replacement=False)
    results = []
    for idx, sample in global_list:
        self.callback_data = [data[i] for i in sample]
        obj, theta = self.theta_est()
        self.callback_data = [data[i] for i in range(len(data)) if i not in sample]
        bootstrap_theta = self.theta_est_bootstrap(bootstrap_samples)
        training, test = self.confidence_region_test(bootstrap_theta, distribution=distribution, alphas=alphas, test_theta_values=theta)
        results.append((sample, test, training))
    self.callback_data = data
    return results