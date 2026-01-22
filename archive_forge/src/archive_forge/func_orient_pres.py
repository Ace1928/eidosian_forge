def orient_pres(isometry):
    """
    >>> M = Manifold('K4a1')
    >>> kinds = {orient_pres(iso) for iso in M.is_isometric_to(M, True)}
    >>> sorted(kinds)
    [False, True]
    """
    return isometry.cusp_maps()[0].det() > 0