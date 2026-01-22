import cupy
def invresz(r, p, k, tol=0.001, rtype='avg'):
    """Compute b(z) and a(z) from partial fraction expansion.

    If `M` is the degree of numerator `b` and `N` the degree of denominator
    `a`::

                b(z)     b[0] + b[1] z**(-1) + ... + b[M] z**(-M)
        H(z) = ------ = ------------------------------------------
                a(z)     a[0] + a[1] z**(-1) + ... + a[N] z**(-N)

    then the partial-fraction expansion H(z) is defined as::

                 r[0]                   r[-1]
         = --------------- + ... + ---------------- + k[0] + k[1]z**(-1) ...
           (1-p[0]z**(-1))         (1-p[-1]z**(-1))

    If there are any repeated roots (closer than `tol`), then the partial
    fraction expansion has terms like::

             r[i]              r[i+1]                    r[i+n-1]
        -------------- + ------------------ + ... + ------------------
        (1-p[i]z**(-1))  (1-p[i]z**(-1))**2         (1-p[i]z**(-1))**n

    This function is used for polynomials in negative powers of z,
    such as digital filters in DSP.  For positive powers, use `invres`.

    Parameters
    ----------
    r : array_like
        Residues corresponding to the poles. For repeated poles, the residues
        must be ordered to correspond to ascending by power fractions.
    p : array_like
        Poles. Equal poles must be adjacent.
    k : array_like
        Coefficients of the direct polynomial term.
    tol : float, optional
        The tolerance for two roots to be considered equal in terms of
        the distance between them. Default is 1e-3. See `unique_roots`
        for further details.
    rtype : {'avg', 'min', 'max'}, optional
        Method for computing a root to represent a group of identical roots.
        Default is 'avg'. See `unique_roots` for further details.

    Returns
    -------
    b : ndarray
        Numerator polynomial coefficients.
    a : ndarray
        Denominator polynomial coefficients.

    See Also
    --------
    scipy.signal.invresz
    residuez, unique_roots, invres

    """
    r = cupy.atleast_1d(r)
    p = cupy.atleast_1d(p)
    k = cupy.trim_zeros(cupy.atleast_1d(k), 'b')
    unique_poles, multiplicity = unique_roots(p, tol, rtype)
    factors, denominator = _compute_factors(unique_poles, multiplicity, include_powers=True)
    if len(k) == 0:
        numerator = 0
    else:
        numerator = cupy.polymul(k[::-1], denominator[::-1])
    for residue, factor in zip(r, factors):
        numerator = cupy.polyadd(numerator, residue * factor[::-1])
    return (numerator[::-1], denominator)