import warnings
import re
import sys
import pickle
from pathlib import Path
import os
import json
import platform
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import numpy
import numpy as np
from numpy import typecodes, array
from numpy.lib.recfunctions import rec_append_fields
from scipy import special
from scipy._lib._util import check_random_state
from scipy.integrate import (IntegrationWarning, quad, trapezoid,
import scipy.stats as stats
from scipy.stats._distn_infrastructure import argsreduce
import scipy.stats.distributions
from scipy.special import xlogy, polygamma, entr
from scipy.stats._distr_params import distcont, invdistcont
from .test_discrete_basic import distdiscrete, invdistdiscrete
from scipy.stats._continuous_distns import FitDataError, _argus_phi
from scipy.optimize import root, fmin, differential_evolution
from itertools import product
def test_invweibull_fit():
    """
    Test fitting invweibull to data.

    Here is a the same calculation in R:

    > library(evd)
    > library(fitdistrplus)
    > x = c(1, 1.25, 2, 2.5, 2.8,  3, 3.8, 4, 5, 8, 10, 12, 64, 99)
    > result = fitdist(x, 'frechet', control=list(reltol=1e-13),
    +                  fix.arg=list(loc=0), start=list(shape=2, scale=3))
    > result
    Fitting of the distribution ' frechet ' by maximum likelihood
    Parameters:
          estimate Std. Error
    shape 1.048482  0.2261815
    scale 3.099456  0.8292887
    Fixed parameters:
        value
    loc     0

    """

    def optimizer(func, x0, args=(), disp=0):
        return fmin(func, x0, args=args, disp=disp, xtol=1e-12, ftol=1e-12)
    x = np.array([1, 1.25, 2, 2.5, 2.8, 3, 3.8, 4, 5, 8, 10, 12, 64, 99])
    c, loc, scale = stats.invweibull.fit(x, floc=0, optimizer=optimizer)
    assert_allclose(c, 1.048482, rtol=5e-06)
    assert loc == 0
    assert_allclose(scale, 3.099456, rtol=5e-06)