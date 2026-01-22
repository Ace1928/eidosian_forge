from numpy.testing import assert_equal
import numpy as np
def dummy_product(d1, d2, method='full'):
    """dummy variable from product of two dummy variables

    Parameters
    ----------
    d1, d2 : ndarray
        two dummy variables, assumes full set for methods 'drop-last'
        and 'drop-first'
    method : {'full', 'drop-last', 'drop-first'}
        'full' returns the full product, encoding of intersection of
        categories.
        The drop methods provide a difference dummy encoding:
        (constant, main effects, interaction effects). The first or last columns
        of the dummy variable (i.e. levels) are dropped to get full rank
        dummy matrix.

    Returns
    -------
    dummy : ndarray
        dummy variable for product, see method

    """
    if method == 'full':
        dd = (d1[:, :, None] * d2[:, None, :]).reshape(d1.shape[0], -1)
    elif method == 'drop-last':
        d12rl = dummy_product(d1[:, :-1], d2[:, :-1])
        dd = np.column_stack((np.ones(d1.shape[0], int), d1[:, :-1], d2[:, :-1], d12rl))
    elif method == 'drop-first':
        d12r = dummy_product(d1[:, 1:], d2[:, 1:])
        dd = np.column_stack((np.ones(d1.shape[0], int), d1[:, 1:], d2[:, 1:], d12r))
    else:
        raise ValueError('method not recognized')
    return dd