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
@pytest.mark.parametrize('case_result', pregenerated_data['pdf_data'])
def test_pdf_against_mp(self, case_result):
    src_case = case_result['src_case']
    mp_result = case_result['mp_result']
    qkv = (src_case['q'], src_case['k'], src_case['v'])
    res = stats.studentized_range.pdf(*qkv)
    assert_allclose(res, mp_result, atol=src_case['expected_atol'], rtol=src_case['expected_rtol'])