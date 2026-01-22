import numpy as np
import scipy.sparse
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.sum import sum as sum_
from cvxpy.atoms.elementwise.maximum import maximum
from cvxpy.expressions.expression import Expression
Elementwise log of the cumulative distribution function of a standard normal random variable.

    The implementation is a quadratic approximation with modest accuracy over [-4, 4].
    For details on the nature of the approximation, refer to
    `CVXPY GitHub PR #1224 <https://github.com/cvxpy/cvxpy/pull/1224#issue-793221374>`_.

    .. note::

        SciPy's analog of ``log_normcdf`` is called `log_ndtr <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.log_ndtr.html>`_.
        We opted not to use that name because its meaning would not be obvious to the casual user.
    