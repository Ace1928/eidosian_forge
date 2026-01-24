from functools import reduce
import math
import numpy as np
from ._stats_py import power_divergence
from ._relative_risk import relative_risk
from ._crosstab import crosstab
from ._odds_ratio import odds_ratio
from scipy._lib._bunch import _make_tuple_bunch
def association(observed, method='cramer', correction=False, lambda_=None):
    """Calculates degree of association between two nominal variables.

    The function provides the option for computing one of three measures of
    association between two nominal variables from the data given in a 2d
    contingency table: Tschuprow's T, Pearson's Contingency Coefficient
    and Cramer's V.

    Parameters
    ----------
    observed : array-like
        The array of observed values
    method : {"cramer", "tschuprow", "pearson"} (default = "cramer")
        The association test statistic.
    correction : bool, optional
        Inherited from `scipy.stats.contingency.chi2_contingency()`
    lambda_ : float or str, optional
        Inherited from `scipy.stats.contingency.chi2_contingency()`

    Returns
    -------
    statistic : float
        Value of the test statistic

    Notes
    -----
    Cramer's V, Tschuprow's T and Pearson's Contingency Coefficient, all
    measure the degree to which two nominal or ordinal variables are related,
    or the level of their association. This differs from correlation, although
    many often mistakenly consider them equivalent. Correlation measures in
    what way two variables are related, whereas, association measures how
    related the variables are. As such, association does not subsume
    independent variables, and is rather a test of independence. A value of
    1.0 indicates perfect association, and 0.0 means the variables have no
    association.

    Both the Cramer's V and Tschuprow's T are extensions of the phi
    coefficient.  Moreover, due to the close relationship between the
    Cramer's V and Tschuprow's T the returned values can often be similar
    or even equivalent.  They are likely to diverge more as the array shape
    diverges from a 2x2.

    References
    ----------
    .. [1] "Tschuprow's T",
           https://en.wikipedia.org/wiki/Tschuprow's_T
    .. [2] Tschuprow, A. A. (1939)
           Principles of the Mathematical Theory of Correlation;
           translated by M. Kantorowitsch. W. Hodge & Co.
    .. [3] "Cramer's V", https://en.wikipedia.org/wiki/Cramer's_V
    .. [4] "Nominal Association: Phi and Cramer's V",
           http://www.people.vcu.edu/~pdattalo/702SuppRead/MeasAssoc/NominalAssoc.html
    .. [5] Gingrich, Paul, "Association Between Variables",
           http://uregina.ca/~gingrich/ch11a.pdf

    Examples
    --------
    An example with a 4x2 contingency table:

    >>> import numpy as np
    >>> from scipy.stats.contingency import association
    >>> obs4x2 = np.array([[100, 150], [203, 322], [420, 700], [320, 210]])

    Pearson's contingency coefficient

    >>> association(obs4x2, method="pearson")
    0.18303298140595667

    Cramer's V

    >>> association(obs4x2, method="cramer")
    0.18617813077483678

    Tschuprow's T

    >>> association(obs4x2, method="tschuprow")
    0.14146478928062995
    """
    arr = np.asarray(observed)
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError('`observed` must be an integer array.')
    if len(arr.shape) != 2:
        raise ValueError('method only accepts 2d arrays')
    chi2_stat = chi2_contingency(arr, correction=correction, lambda_=lambda_)
    phi2 = chi2_stat.statistic / arr.sum()
    n_rows, n_cols = arr.shape
    if method == 'cramer':
        value = phi2 / min(n_cols - 1, n_rows - 1)
    elif method == 'tschuprow':
        value = phi2 / math.sqrt((n_rows - 1) * (n_cols - 1))
    elif method == 'pearson':
        value = phi2 / (1 + phi2)
    else:
        raise ValueError("Invalid argument value: 'method' argument must be 'cramer', 'tschuprow', or 'pearson'")
    return math.sqrt(value)