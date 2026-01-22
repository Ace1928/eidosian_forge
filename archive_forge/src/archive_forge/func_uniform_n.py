from ...context import current_context
from ..numpy import _internal as _npi
def uniform_n(low=0.0, high=1.0, batch_shape=None, dtype=None, ctx=None):
    """Draw samples from a uniform distribution.

    Samples are uniformly distributed over the half-open interval
    ``[low, high)`` (includes low, but excludes high).  In other words,
    any value within the given interval is equally likely to be drawn
    by `uniform`.

    Parameters
    ----------
    low : float, ndarray, optional
        Lower boundary of the output interval.  All values generated will be
        greater than or equal to low.  The default value is 0.
    high : float, ndarray, optional
        Upper boundary of the output interval.  All values generated will be
        less than high.  The default value is 1.0.
    batch_shape : int or tuple of ints, optional
        Batch shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k * broadcast(low, high).size`` samples are drawn.
        If size is ``None`` (default),
        a scalar tensor containing a single value is returned if
        ``low`` and ``high`` are both scalars. Otherwise,
        ``np.broadcast(low, high).size`` samples are drawn.
    dtype : {'float16', 'float32', 'float64'}, optional
        Data type of output samples. Default is 'float32'
    ctx : Context, optional
        Device context of output. Default is current context.

    Returns
    -------
    out : ndarray
        Drawn samples from the parameterized uniform distribution.

    See Also
    --------
    randint : Discrete uniform distribution, yielding integers.
    rand : Convenience function that accepts dimensions as input, e.g.,
           ``rand(2,2)`` would generate a 2-by-2 array of floats,
           uniformly distributed over ``[0, 1)``.

    Notes
    -----
    The probability density function of the uniform distribution is

    .. math:: p(x) = \\frac{1}{b - a}

    anywhere within the interval ``[a, b)``, and zero elsewhere.

    When ``high`` == ``low``, values of ``low`` will be returned.
    If ``high`` < ``low``, the results are officially undefined
    and may eventually raise an error, i.e. do not rely on this
    function to behave when passed arguments satisfying that
    inequality condition.
    """
    from ...numpy import ndarray as np_ndarray
    input_type = (isinstance(low, np_ndarray), isinstance(high, np_ndarray))
    if dtype is None:
        dtype = 'float32'
    if ctx is None:
        ctx = current_context()
    if batch_shape == ():
        batch_shape = None
    if input_type == (True, True):
        return _npi.uniform_n(low, high, low=None, high=None, size=batch_shape, ctx=ctx, dtype=dtype)
    elif input_type == (False, True):
        return _npi.uniform_n(high, low=low, high=None, size=batch_shape, ctx=ctx, dtype=dtype)
    elif input_type == (True, False):
        return _npi.uniform_n(low, low=None, high=high, size=batch_shape, ctx=ctx, dtype=dtype)
    else:
        return _npi.uniform_n(low=low, high=high, size=batch_shape, ctx=ctx, dtype=dtype)