from numpy.testing import assert_equal
import numpy as np
def contrast_product(names1, names2, intgroup1=None, intgroup2=None, pairs=False):
    """build contrast matrices for products of two categorical variables

    this is an experimental script and should be converted to a class

    Parameters
    ----------
    names1, names2 : lists of strings
        contains the list of level labels for each categorical variable
    intgroup1, intgroup2 : ndarrays     TODO: this part not tested, finished yet
        categorical variable


    Notes
    -----
    This creates a full rank matrix. It does not do all pairwise comparisons,
    parameterization is using contrast_all_one to get differences with first
    level.

    ? does contrast_all_pairs work as a plugin to get all pairs ?

    """
    n1 = len(names1)
    n2 = len(names2)
    names_prod = ['{}_{}'.format(i, j) for i in names1 for j in names2]
    ee1 = np.zeros((1, n1))
    ee1[0, 0] = 1
    if not pairs:
        dd = np.r_[ee1, -contrast_all_one(n1)]
    else:
        dd = np.r_[ee1, -contrast_allpairs(n1)]
    contrast_prod = np.kron(dd[1:], np.eye(n2))
    names_contrast_prod0 = contrast_labels(contrast_prod, names_prod, reverse=True)
    names_contrast_prod = [''.join(['{}{}'.format(signstr(c, noplus=True), v) for c, v in zip(row, names_prod)[::-1] if c != 0]) for row in contrast_prod]
    ee2 = np.zeros((1, n2))
    ee2[0, 0] = 1
    if not pairs:
        dd2 = np.r_[ee2, -contrast_all_one(n2)]
    else:
        dd2 = np.r_[ee2, -contrast_allpairs(n2)]
    contrast_prod2 = np.kron(np.eye(n1), dd2[1:])
    names_contrast_prod2 = [''.join(['{}{}'.format(signstr(c, noplus=True), v) for c, v in zip(row, names_prod)[::-1] if c != 0]) for row in contrast_prod2]
    if intgroup1 is not None and intgroup1 is not None:
        d1, _ = dummy_1d(intgroup1)
        d2, _ = dummy_1d(intgroup2)
        dummy = dummy_product(d1, d2)
    else:
        dummy = None
    return (names_prod, contrast_prod, names_contrast_prod, contrast_prod2, names_contrast_prod2, dummy)