from scipy import stats, integrate, special
import numpy as np
def inner_cont(polys, lower, upper, weight=None):
    """inner product of continuous function (with weight=1)

    Parameters
    ----------
    polys : list of callables
        polynomial instances
    lower : float
        lower integration limit
    upper : float
        upper integration limit
    weight : callable or None
        weighting function

    Returns
    -------
    innp : ndarray
        symmetric 2d square array with innerproduct of all function pairs
    err : ndarray
        numerical error estimate from scipy.integrate.quad, same dimension as innp

    Examples
    --------
    >>> from scipy.special import chebyt
    >>> polys = [chebyt(i) for i in range(4)]
    >>> r, e = inner_cont(polys, -1, 1)
    >>> r
    array([[ 2.        ,  0.        , -0.66666667,  0.        ],
           [ 0.        ,  0.66666667,  0.        , -0.4       ],
           [-0.66666667,  0.        ,  0.93333333,  0.        ],
           [ 0.        , -0.4       ,  0.        ,  0.97142857]])

    """
    n_polys = len(polys)
    innerprod = np.empty((n_polys, n_polys))
    innerprod.fill(np.nan)
    interr = np.zeros((n_polys, n_polys))
    for i in range(n_polys):
        for j in range(i + 1):
            p1 = polys[i]
            p2 = polys[j]
            if weight is not None:
                innp, err = integrate.quad(lambda x: p1(x) * p2(x) * weight(x), lower, upper)
            else:
                innp, err = integrate.quad(lambda x: p1(x) * p2(x), lower, upper)
            innerprod[i, j] = innp
            interr[i, j] = err
            if not i == j:
                innerprod[j, i] = innp
                interr[j, i] = err
    return (innerprod, interr)