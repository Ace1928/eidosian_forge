import numpy
def chi_square_test(observed, expected, alpha=0.05, df=None):
    """Testing Goodness-of-fit Test with Pearson's Chi-squared Test.

    Args:
        observed (list of ints): List of # of counts each element is observed.
        expected (list of floats): List of # of counts each element is expected
            to be observed.
        alpha (float): Significance level. Currently,
            only 0.05 and 0.01 are acceptable.
        df (int): Degree of freedom. If ``None``,
            it is set to the length of ``observed`` minus 1.

    Returns:
        bool: ``True`` if null hypothesis is **NOT** reject.
        Otherwise, ``False``.
    """
    if df is None:
        df = observed.size - 1
    if alpha == 0.01:
        alpha_idx = 0
    elif alpha == 0.05:
        alpha_idx = 1
    else:
        raise ValueError('support only alpha == 0.05 or 0.01')
    chi_square = numpy.sum((observed - expected) ** 2 / expected)
    return chi_square < chi_square_table[alpha_idx][df]