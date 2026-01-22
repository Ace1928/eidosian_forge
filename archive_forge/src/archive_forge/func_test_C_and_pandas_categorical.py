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
def test_C_and_pandas_categorical():
    if not have_pandas_categorical:
        return
    objs = [pandas_Categorical_from_codes([1, 0, 1], ['b', 'a'])]
    if have_pandas_categorical_dtype:
        objs.append(pandas.Series(objs[0]))
    for obj in objs:
        d = {'obj': obj}
        assert np.allclose(dmatrix('obj', d), [[1, 1], [1, 0], [1, 1]])
        assert np.allclose(dmatrix('C(obj)', d), [[1, 1], [1, 0], [1, 1]])
        assert np.allclose(dmatrix("C(obj, levels=['b', 'a'])", d), [[1, 1], [1, 0], [1, 1]])
        assert np.allclose(dmatrix("C(obj, levels=['a', 'b'])", d), [[1, 0], [1, 1], [1, 0]])