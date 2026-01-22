import cupy
def unique_roots(p, tol=0.001, rtype='min'):
    """Determine unique roots and their multiplicities from a list of roots.

    Parameters
    ----------
    p : array_like
        The list of roots.
    tol : float, optional
        The tolerance for two roots to be considered equal in terms of
        the distance between them. Default is 1e-3. Refer to Notes about
        the details on roots grouping.
    rtype : {'max', 'maximum', 'min', 'minimum', 'avg', 'mean'}, optional
        How to determine the returned root if multiple roots are within
        `tol` of each other.

          - 'max', 'maximum': pick the maximum of those roots
          - 'min', 'minimum': pick the minimum of those roots
          - 'avg', 'mean': take the average of those roots

        When finding minimum or maximum among complex roots they are compared
        first by the real part and then by the imaginary part.

    Returns
    -------
    unique : ndarray
        The list of unique roots.
    multiplicity : ndarray
        The multiplicity of each root.

    See Also
    --------
    scipy.signal.unique_roots

    Notes
    -----
    If we have 3 roots ``a``, ``b`` and ``c``, such that ``a`` is close to
    ``b`` and ``b`` is close to ``c`` (distance is less than `tol`), then it
    doesn't necessarily mean that ``a`` is close to ``c``. It means that roots
    grouping is not unique. In this function we use "greedy" grouping going
    through the roots in the order they are given in the input `p`.

    This utility function is not specific to roots but can be used for any
    sequence of values for which uniqueness and multiplicity has to be
    determined. For a more general routine, see `numpy.unique`.

    """
    if rtype in ['max', 'maximum']:
        reduce = cupy.max
    elif rtype in ['min', 'minimum']:
        reduce = cupy.min
    elif rtype in ['avg', 'mean']:
        reduce = cupy.mean
    else:
        raise ValueError("`rtype` must be one of {'max', 'maximum', 'min', 'minimum', 'avg', 'mean'}")
    points = cupy.empty((p.shape[0], 2))
    points[:, 0] = cupy.real(p)
    points[:, 1] = cupy.imag(p)
    dist = cupy.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    p_unique = []
    p_multiplicity = []
    used = cupy.zeros(p.shape[0], dtype=bool)
    for i, ds in enumerate(dist):
        if used[i]:
            continue
        mask = (ds < tol) & ~used
        group = ds[mask]
        if group.size > 0:
            p_unique.append(reduce(p[mask]))
            p_multiplicity.append(group.shape[0])
        used[mask] = True
    return (cupy.asarray(p_unique), cupy.asarray(p_multiplicity))