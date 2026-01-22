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
def test_dmatrix_NA_action():
    data = {'x': [1, 2, 3, np.nan], 'y': [np.nan, 20, 30, 40]}
    return_types = ['matrix']
    if have_pandas:
        return_types.append('dataframe')
    for return_type in return_types:
        mat = dmatrix('x + y', data=data, return_type=return_type)
        assert np.array_equal(mat, [[1, 2, 20], [1, 3, 30]])
        if return_type == 'dataframe':
            assert mat.index.equals(pandas.Index([1, 2]))
        pytest.raises(PatsyError, dmatrix, 'x + y', data=data, return_type=return_type, NA_action='raise')
        lmat, rmat = dmatrices('y ~ x', data=data, return_type=return_type)
        assert np.array_equal(lmat, [[20], [30]])
        assert np.array_equal(rmat, [[1, 2], [1, 3]])
        if return_type == 'dataframe':
            assert lmat.index.equals(pandas.Index([1, 2]))
            assert rmat.index.equals(pandas.Index([1, 2]))
        pytest.raises(PatsyError, dmatrices, 'y ~ x', data=data, return_type=return_type, NA_action='raise')
        lmat, rmat = dmatrices('y ~ 1', data=data, return_type=return_type)
        assert np.array_equal(lmat, [[20], [30], [40]])
        assert np.array_equal(rmat, [[1], [1], [1]])
        if return_type == 'dataframe':
            assert lmat.index.equals(pandas.Index([1, 2, 3]))
            assert rmat.index.equals(pandas.Index([1, 2, 3]))
        pytest.raises(PatsyError, dmatrices, 'y ~ 1', data=data, return_type=return_type, NA_action='raise')