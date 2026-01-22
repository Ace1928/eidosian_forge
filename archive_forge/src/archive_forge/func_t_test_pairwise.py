import numpy as np
from scipy.stats import f as fdist
from scipy.stats import t as student_t
from scipy import stats
from statsmodels.tools.tools import clean0, fullrank
from statsmodels.stats.multitest import multipletests
def t_test_pairwise(result, term_name, method='hs', alpha=0.05, factor_labels=None, ignore=False):
    """
    Perform pairwise t_test with multiple testing corrected p-values.

    This uses the formula design_info encoding contrast matrix and should
    work for all encodings of a main effect.

    Parameters
    ----------
    result : result instance
        The results of an estimated model with a categorical main effect.
    term_name : str
        name of the term for which pairwise comparisons are computed.
        Term names for categorical effects are created by patsy and
        correspond to the main part of the exog names.
    method : {str, list[str]}
        multiple testing p-value correction, default is 'hs',
        see stats.multipletesting
    alpha : float
        significance level for multiple testing reject decision.
    factor_labels : {list[str], None}
        Labels for the factor levels used for pairwise labels. If not
        provided, then the labels from the formula design_info are used.
    ignore : bool
        Turn off some of the exceptions raised by input checks.

    Returns
    -------
    MultiCompResult
        The results are stored as attributes, the main attributes are the
        following two. Other attributes are added for debugging purposes
        or as background information.

        - result_frame : pandas DataFrame with t_test results and multiple
          testing corrected p-values.
        - contrasts : matrix of constraints of the null hypothesis in the
          t_test.

    Notes
    -----

    Status: experimental. Currently only checked for treatment coding with
    and without specified reference level.

    Currently there are no multiple testing corrected confidence intervals
    available.
    """
    desinfo = result.model.data.design_info
    term_idx = desinfo.term_names.index(term_name)
    term = desinfo.terms[term_idx]
    idx_start = desinfo.term_slices[term].start
    if not ignore and len(term.factors) > 1:
        raise ValueError('interaction effects not yet supported')
    factor = term.factors[0]
    cat = desinfo.factor_infos[factor].categories
    if factor_labels is not None:
        if len(factor_labels) == len(cat):
            cat = factor_labels
        else:
            raise ValueError('factor_labels has the wrong length, should be %d' % len(cat))
    k_level = len(cat)
    cm = desinfo.term_codings[term][0].contrast_matrices[factor].matrix
    k_params = len(result.params)
    labels = _get_pairs_labels(k_level, cat)
    import statsmodels.sandbox.stats.multicomp as mc
    c_all_pairs = -mc.contrast_allpairs(k_level)
    contrasts_sub = c_all_pairs.dot(cm)
    contrasts = _embed_constraints(contrasts_sub, k_params, idx_start)
    res_df = t_test_multi(result, contrasts, method=method, ci_method=None, alpha=alpha, contrast_names=labels)
    res = MultiCompResult(result_frame=res_df, contrasts=contrasts, term=term, contrast_labels=labels, term_encoding_matrix=cm)
    return res