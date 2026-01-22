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
@pytest.mark.parametrize('chi, expected', [(0.5, (0.28414073302940573, 1.2742227939992954, 1.2381254688255896)), (0.2, (0.296172952995264, 1.2951290588110516, 1.1865767100877576)), (0.1, (0.29791447523536274, 1.29806307956989, 1.1793168289857412)), (0.01, (0.2984904104866452, 1.2990283628160553, 1.1769268414080531)), (0.001, (0.298496172925224, 1.2990380082487925, 1.176902956021053)), (0.0001, (0.29849623054991836, 1.2990381047023793, 1.1769027171686324)), (1e-06, (0.2984962311319278, 1.2990381056765605, 1.1769027147562232)), (1e-09, (0.298496231131986, 1.299038105676658, 1.1769027147559818))])
def test_pdf_small_chi(self, chi, expected):
    x = np.array([0.1, 0.5, 0.9])
    assert_allclose(stats.argus.pdf(x, chi), expected, rtol=1e-13)