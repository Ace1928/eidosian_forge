from numbers import Number, Integral
from .. import _api_internal
Construct a constant value for a given type.

    Parameters
    ----------
    value : int or float
        The input value

    dtype : str or None, optional
        The data type.

    Returns
    -------
    expr : Expr
        Constant expression corresponds to the value.
    