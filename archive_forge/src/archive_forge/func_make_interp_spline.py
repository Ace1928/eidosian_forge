import operator
from numpy.core.multiarray import normalize_axis_index
import cupy
from cupyx.scipy import sparse
from cupyx.scipy.sparse.linalg import spsolve
from cupyx.scipy.interpolate._bspline import (
def make_interp_spline(x, y, k=3, t=None, bc_type=None, axis=0, check_finite=True):
    """Compute the (coefficients of) interpolating B-spline.

    Parameters
    ----------
    x : array_like, shape (n,)
        Abscissas.
    y : array_like, shape (n, ...)
        Ordinates.
    k : int, optional
        B-spline degree. Default is cubic, ``k = 3``.
    t : array_like, shape (nt + k + 1,), optional.
        Knots.
        The number of knots needs to agree with the number of data points and
        the number of derivatives at the edges. Specifically, ``nt - n`` must
        equal ``len(deriv_l) + len(deriv_r)``.
    bc_type : 2-tuple or None
        Boundary conditions.
        Default is None, which means choosing the boundary conditions
        automatically. Otherwise, it must be a length-two tuple where the first
        element (``deriv_l``) sets the boundary conditions at ``x[0]`` and
        the second element (``deriv_r``) sets the boundary conditions at
        ``x[-1]``. Each of these must be an iterable of pairs
        ``(order, value)`` which gives the values of derivatives of specified
        orders at the given edge of the interpolation interval.
        Alternatively, the following string aliases are recognized:

        * ``"clamped"``: The first derivatives at the ends are zero. This is
           equivalent to ``bc_type=([(1, 0.0)], [(1, 0.0)])``.
        * ``"natural"``: The second derivatives at ends are zero. This is
          equivalent to ``bc_type=([(2, 0.0)], [(2, 0.0)])``.
        * ``"not-a-knot"`` (default): The first and second segments are the
          same polynomial. This is equivalent to having ``bc_type=None``.
        * ``"periodic"``: The values and the first ``k-1`` derivatives at the
          ends are equivalent.

    axis : int, optional
        Interpolation axis. Default is 0.
    check_finite : bool, optional
        Whether to check that the input arrays contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default is True.

    Returns
    -------
    b : a BSpline object of the degree ``k`` and with knots ``t``.

    """
    if bc_type is None or bc_type == 'not-a-knot' or bc_type == 'periodic':
        deriv_l, deriv_r = (None, None)
    elif isinstance(bc_type, str):
        deriv_l, deriv_r = (bc_type, bc_type)
    else:
        try:
            deriv_l, deriv_r = bc_type
        except TypeError as e:
            raise ValueError('Unknown boundary condition: %s' % bc_type) from e
    y = cupy.asarray(y)
    axis = normalize_axis_index(axis, y.ndim)
    x = _as_float_array(x, check_finite)
    y = _as_float_array(y, check_finite)
    y = cupy.moveaxis(y, axis, 0)
    if bc_type == 'periodic' and (not cupy.allclose(y[0], y[-1], atol=1e-15)):
        raise ValueError('First and last points does not match while periodic case expected')
    if x.size != y.shape[0]:
        raise ValueError('Shapes of x {} and y {} are incompatible'.format(x.shape, y.shape))
    if (x[1:] == x[:-1]).any():
        raise ValueError('Expect x to not have duplicates')
    if x.ndim != 1 or (x[1:] < x[:-1]).any():
        raise ValueError('Expect x to be a 1D strictly increasing sequence.')
    if k == 0:
        if any((_ is not None for _ in (t, deriv_l, deriv_r))):
            raise ValueError('Too much info for k=0: t and bc_type can only be None.')
        t = cupy.r_[x, x[-1]]
        c = cupy.asarray(y)
        c = cupy.ascontiguousarray(c, dtype=_get_dtype(c.dtype))
        return BSpline.construct_fast(t, c, k, axis=axis)
    if k == 1 and t is None:
        if not (deriv_l is None and deriv_r is None):
            raise ValueError('Too much info for k=1: bc_type can only be None.')
        t = cupy.r_[x[0], x, x[-1]]
        c = cupy.asarray(y)
        c = cupy.ascontiguousarray(c, dtype=_get_dtype(c.dtype))
        return BSpline.construct_fast(t, c, k, axis=axis)
    k = operator.index(k)
    if bc_type == 'periodic' and t is not None:
        raise NotImplementedError('For periodic case t is constructed automatically and can not be passed manually')
    if t is None:
        if deriv_l is None and deriv_r is None:
            if bc_type == 'periodic':
                t = _periodic_knots(x, k)
            elif k == 2:
                t = (x[1:] + x[:-1]) / 2.0
                t = cupy.r_[(x[0],) * (k + 1), t[1:-1], (x[-1],) * (k + 1)]
            else:
                t = _not_a_knot(x, k)
        else:
            t = _augknt(x, k)
    t = _as_float_array(t, check_finite)
    if k < 0:
        raise ValueError('Expect non-negative k.')
    if t.ndim != 1 or (t[1:] < t[:-1]).any():
        raise ValueError('Expect t to be a 1-D sorted array_like.')
    if t.size < x.size + k + 1:
        raise ValueError('Got %d knots, need at least %d.' % (t.size, x.size + k + 1))
    if x[0] < t[k] or x[-1] > t[-k]:
        raise ValueError('Out of bounds w/ x = %s.' % x)
    if bc_type == 'periodic':
        return _make_periodic_spline(x, y, t, k, axis)
    deriv_l = _convert_string_aliases(deriv_l, y.shape[1:])
    deriv_l_ords, deriv_l_vals = _process_deriv_spec(deriv_l)
    nleft = deriv_l_ords.shape[0]
    deriv_r = _convert_string_aliases(deriv_r, y.shape[1:])
    deriv_r_ords, deriv_r_vals = _process_deriv_spec(deriv_r)
    nright = deriv_r_ords.shape[0]
    n = x.size
    nt = t.size - k - 1
    if nt - n != nleft + nright:
        raise ValueError('The number of derivatives at boundaries does not match: expected %s, got %s + %s' % (nt - n, nleft, nright))
    if y.size == 0:
        c = cupy.zeros((nt,) + y.shape[1:], dtype=float)
        return BSpline.construct_fast(t, c, k, axis=axis)
    matr = BSpline.design_matrix(x, t, k)
    if nleft > 0 or nright > 0:
        temp = cupy.zeros((nt,), dtype=float)
        num_c = 1
        dummy_c = cupy.empty((nt, num_c), dtype=float)
        out = cupy.empty((1, 1), dtype=dummy_c.dtype)
        d_boor_kernel = _get_module_func(D_BOOR_MODULE, 'd_boor', dummy_c)
        intervals_bc = cupy.empty(2, dtype=cupy.int64)
        interval_kernel = _get_module_func(INTERVAL_MODULE, 'find_interval')
        interval_kernel((1,), (2,), (t, cupy.r_[x[0], x[-1]], intervals_bc, k, nt, False, 2))
    if nleft > 0:
        x0 = cupy.array([x[0]], dtype=x.dtype)
        rows = cupy.zeros((nleft, nt), dtype=float)
        for j, m in enumerate(deriv_l_ords):
            d_boor_kernel((1,), (1,), (t, dummy_c, k, int(m), x0, intervals_bc, out, temp, num_c, 0, 1))
            left = intervals_bc[0]
            rows[j, left - k:left + 1] = temp[:k + 1]
        matr = sparse.vstack([sparse.csr_matrix(rows), matr])
    if nright > 0:
        intervals_bc[0] = intervals_bc[-1]
        x0 = cupy.array([x[-1]], dtype=x.dtype)
        rows = cupy.zeros((nright, nt), dtype=float)
        for j, m in enumerate(deriv_r_ords):
            d_boor_kernel((1,), (1,), (t, dummy_c, k, int(m), x0, intervals_bc, out, temp, num_c, 0, 1))
            left = intervals_bc[0]
            rows[j, left - k:left + 1] = temp[:k + 1]
        matr = sparse.vstack([matr, sparse.csr_matrix(rows)])
    extradim = prod(y.shape[1:])
    rhs = cupy.empty((nt, extradim), dtype=y.dtype)
    if nleft > 0:
        rhs[:nleft] = deriv_l_vals.reshape(-1, extradim)
    rhs[nleft:nt - nright] = y.reshape(-1, extradim)
    if nright > 0:
        rhs[nt - nright:] = deriv_r_vals.reshape(-1, extradim)
    if cupy.issubdtype(rhs.dtype, cupy.complexfloating):
        coef = spsolve(matr, rhs.real) + spsolve(matr, rhs.imag) * 1j
    else:
        coef = spsolve(matr, rhs)
    coef = cupy.ascontiguousarray(coef.reshape((nt,) + y.shape[1:]))
    return BSpline(t, coef, k)