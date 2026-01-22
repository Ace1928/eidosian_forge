import sys
import __future__
import six
import numpy as np
import pytest
from patsy import PatsyError
from patsy.design_info import DesignMatrix, DesignInfo
from patsy.eval import EvalEnvironment
from patsy.desc import ModelDesc, Term, INTERCEPT
from patsy.categorical import C
from patsy.contrasts import Helmert
from patsy.user_util import balanced, LookupFactor
from patsy.build import (design_matrix_builders,
from patsy.highlevel import *
from patsy.util import (have_pandas,
from patsy.origin import Origin
def test_env_transform():
    t('~ np.sin(x)', {'x': [1, 2, 3]}, 0, True, [[1, np.sin(1)], [1, np.sin(2)], [1, np.sin(3)]], ['Intercept', 'np.sin(x)'])