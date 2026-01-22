import numpy as np
import numpy.linalg as npl
def nearest_pos_semi_def(B):
    """Least squares positive semi-definite tensor estimation

    Reference: Niethammer M, San Jose Estepar R, Bouix S, Shenton M,
    Westin CF.  On diffusion tensor estimation. Conf Proc IEEE Eng Med
    Biol Soc.  2006;1:2622-5. PubMed PMID: 17946125; PubMed Central
    PMCID: PMC2791793.

    Parameters
    ----------
    B : (3,3) array-like
       B matrix - symmetric. We do not check the symmetry.

    Returns
    -------
    npds : (3,3) array
       Estimated nearest positive semi-definite array to matrix `B`.

    Examples
    --------
    >>> B = np.diag([1, 1, -1])
    >>> nearest_pos_semi_def(B)
    array([[0.75, 0.  , 0.  ],
           [0.  , 0.75, 0.  ],
           [0.  , 0.  , 0.  ]])
    """
    B = np.asarray(B)
    vals, vecs = npl.eigh(B)
    inds = np.argsort(vals)[::-1]
    vals = vals[inds]
    cardneg = np.sum(vals < 0)
    if cardneg == 0:
        return B
    if cardneg == 3:
        return np.zeros((3, 3))
    lam1a, lam2a, lam3a = vals
    scalers = np.zeros((3,))
    if cardneg == 2:
        b112 = np.max([0, lam1a + (lam2a + lam3a) / 3.0])
        scalers[0] = b112
    elif cardneg == 1:
        lam1b = lam1a + 0.25 * lam3a
        lam2b = lam2a + 0.25 * lam3a
        if lam1b >= 0 and lam2b >= 0:
            scalers[:2] = (lam1b, lam2b)
        else:
            if lam2b < 0:
                b111 = np.max([0, lam1a + (lam2a + lam3a) / 3.0])
                scalers[0] = b111
            if lam1b < 0:
                b221 = np.max([0, lam2a + (lam1a + lam3a) / 3.0])
                scalers[1] = b221
    scalers = scalers[np.argsort(inds)]
    return np.dot(vecs, np.dot(np.diag(scalers), vecs.T))