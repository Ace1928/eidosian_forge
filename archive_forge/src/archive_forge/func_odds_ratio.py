import numpy as np
from scipy.special import ndtri
from scipy.optimize import brentq
from ._discrete_distns import nchypergeom_fisher
from ._common import ConfidenceInterval
def odds_ratio(table, *, kind='conditional'):
    """
    Compute the odds ratio for a 2x2 contingency table.

    Parameters
    ----------
    table : array_like of ints
        A 2x2 contingency table.  Elements must be non-negative integers.
    kind : str, optional
        Which kind of odds ratio to compute, either the sample
        odds ratio (``kind='sample'``) or the conditional odds ratio
        (``kind='conditional'``).  Default is ``'conditional'``.

    Returns
    -------
    result : `~scipy.stats._result_classes.OddsRatioResult` instance
        The returned object has two computed attributes:

        statistic : float
            * If `kind` is ``'sample'``, this is sample (or unconditional)
              estimate, given by
              ``table[0, 0]*table[1, 1]/(table[0, 1]*table[1, 0])``.
            * If `kind` is ``'conditional'``, this is the conditional
              maximum likelihood estimate for the odds ratio. It is
              the noncentrality parameter of Fisher's noncentral
              hypergeometric distribution with the same hypergeometric
              parameters as `table` and whose mean is ``table[0, 0]``.

        The object has the method `confidence_interval` that computes
        the confidence interval of the odds ratio.

    See Also
    --------
    scipy.stats.fisher_exact
    relative_risk

    Notes
    -----
    The conditional odds ratio was discussed by Fisher (see "Example 1"
    of [1]_).  Texts that cover the odds ratio include [2]_ and [3]_.

    .. versionadded:: 1.10.0

    References
    ----------
    .. [1] R. A. Fisher (1935), The logic of inductive inference,
           Journal of the Royal Statistical Society, Vol. 98, No. 1,
           pp. 39-82.
    .. [2] Breslow NE, Day NE (1980). Statistical methods in cancer research.
           Volume I - The analysis of case-control studies. IARC Sci Publ.
           (32):5-338. PMID: 7216345. (See section 4.2.)
    .. [3] H. Sahai and A. Khurshid (1996), Statistics in Epidemiology:
           Methods, Techniques, and Applications, CRC Press LLC, Boca
           Raton, Florida.
    .. [4] Berger, Jeffrey S. et al. "Aspirin for the Primary Prevention of
           Cardiovascular Events in Women and Men: A Sex-Specific
           Meta-analysis of Randomized Controlled Trials."
           JAMA, 295(3):306-313, :doi:`10.1001/jama.295.3.306`, 2006.

    Examples
    --------
    In epidemiology, individuals are classified as "exposed" or
    "unexposed" to some factor or treatment. If the occurrence of some
    illness is under study, those who have the illness are often
    classified as "cases", and those without it are "noncases".  The
    counts of the occurrences of these classes gives a contingency
    table::

                    exposed    unexposed
        cases          a           b
        noncases       c           d

    The sample odds ratio may be written ``(a/c) / (b/d)``.  ``a/c`` can
    be interpreted as the odds of a case occurring in the exposed group,
    and ``b/d`` as the odds of a case occurring in the unexposed group.
    The sample odds ratio is the ratio of these odds.  If the odds ratio
    is greater than 1, it suggests that there is a positive association
    between being exposed and being a case.

    Interchanging the rows or columns of the contingency table inverts
    the odds ratio, so it is import to understand the meaning of labels
    given to the rows and columns of the table when interpreting the
    odds ratio.

    In [4]_, the use of aspirin to prevent cardiovascular events in women
    and men was investigated. The study notably concluded:

        ...aspirin therapy reduced the risk of a composite of
        cardiovascular events due to its effect on reducing the risk of
        ischemic stroke in women [...]

    The article lists studies of various cardiovascular events. Let's
    focus on the ischemic stoke in women.

    The following table summarizes the results of the experiment in which
    participants took aspirin or a placebo on a regular basis for several
    years. Cases of ischemic stroke were recorded::

                          Aspirin   Control/Placebo
        Ischemic stroke     176           230
        No stroke         21035         21018

    The question we ask is "Is there evidence that the aspirin reduces the
    risk of ischemic stroke?"

    Compute the odds ratio:

    >>> from scipy.stats.contingency import odds_ratio
    >>> res = odds_ratio([[176, 230], [21035, 21018]])
    >>> res.statistic
    0.7646037659999126

    For this sample, the odds of getting an ischemic stroke for those who have
    been taking aspirin are 0.76 times that of those
    who have received the placebo.

    To make statistical inferences about the population under study,
    we can compute the 95% confidence interval for the odds ratio:

    >>> res.confidence_interval(confidence_level=0.95)
    ConfidenceInterval(low=0.6241234078749812, high=0.9354102892100372)

    The 95% confidence interval for the conditional odds ratio is
    approximately (0.62, 0.94).

    The fact that the entire 95% confidence interval falls below 1 supports
    the authors' conclusion that the aspirin was associated with a
    statistically significant reduction in ischemic stroke.
    """
    if kind not in ['conditional', 'sample']:
        raise ValueError("`kind` must be 'conditional' or 'sample'.")
    c = np.asarray(table)
    if c.shape != (2, 2):
        raise ValueError(f'Invalid shape {c.shape}. The input `table` must be of shape (2, 2).')
    if not np.issubdtype(c.dtype, np.integer):
        raise ValueError(f'`table` must be an array of integers, but got type {c.dtype}')
    c = c.astype(np.int64)
    if np.any(c < 0):
        raise ValueError('All values in `table` must be nonnegative.')
    if 0 in c.sum(axis=0) or 0 in c.sum(axis=1):
        result = OddsRatioResult(_table=c, _kind=kind, statistic=np.nan)
        return result
    if kind == 'sample':
        oddsratio = _sample_odds_ratio(c)
    else:
        oddsratio = _conditional_oddsratio(c)
    result = OddsRatioResult(_table=c, _kind=kind, statistic=oddsratio)
    return result