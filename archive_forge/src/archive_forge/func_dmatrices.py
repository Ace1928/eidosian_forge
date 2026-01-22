import six
import numpy as np
from patsy import PatsyError
from patsy.design_info import DesignMatrix, DesignInfo
from patsy.eval import EvalEnvironment
from patsy.desc import ModelDesc
from patsy.build import (design_matrix_builders,
from patsy.util import (have_pandas, asarray_or_pandas,
def dmatrices(formula_like, data={}, eval_env=0, NA_action='drop', return_type='matrix'):
    """Construct two design matrices given a formula_like and data.

    This function is identical to :func:`dmatrix`, except that it requires
    (and returns) two matrices instead of one. By convention, the first matrix
    is the "outcome" or "y" data, and the second is the "predictor" or "x"
    data.

    See :func:`dmatrix` for details.
    """
    eval_env = EvalEnvironment.capture(eval_env, reference=1)
    lhs, rhs = _do_highlevel_design(formula_like, data, eval_env, NA_action, return_type)
    if lhs.shape[1] == 0:
        raise PatsyError('model is missing required outcome variables')
    return (lhs, rhs)