from __future__ import annotations
from statsmodels.compat.python import lzip
from functools import reduce
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.base.data import handle_data
from statsmodels.base.optimizer import Optimizer
import statsmodels.base.wrapper as wrap
from statsmodels.formula import handle_formula_data
from statsmodels.stats.contrast import (
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.decorators import (
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import nan_dot, recipr
from statsmodels.tools.validation import bool_like
def wald_test_terms(self, skip_single=False, extra_constraints=None, combine_terms=None, scalar=None):
    """
        Compute a sequence of Wald tests for terms over multiple columns.

        This computes joined Wald tests for the hypothesis that all
        coefficients corresponding to a `term` are zero.
        `Terms` are defined by the underlying formula or by string matching.

        Parameters
        ----------
        skip_single : bool
            If true, then terms that consist only of a single column and,
            therefore, refers only to a single parameter is skipped.
            If false, then all terms are included.
        extra_constraints : ndarray
            Additional constraints to test. Note that this input has not been
            tested.
        combine_terms : {list[str], None}
            Each string in this list is matched to the name of the terms or
            the name of the exogenous variables. All columns whose name
            includes that string are combined in one joint test.
        scalar : bool, optional
            Flag indicating whether the Wald test statistic should be returned
            as a sclar float. The current behavior is to return an array.
            This will switch to a scalar float after 0.14 is released. To
            get the future behavior now, set scalar to True. To silence
            the warning and retain the legacy behavior, set scalar to
            False.

        Returns
        -------
        WaldTestResults
            The result instance contains `table` which is a pandas DataFrame
            with the test results: test statistic, degrees of freedom and
            pvalues.

        Examples
        --------
        >>> res_ols = ols("np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)", data).fit()
        >>> res_ols.wald_test_terms()
        <class 'statsmodels.stats.contrast.WaldTestResults'>
                                                  F                P>F  df constraint  df denom
        Intercept                        279.754525  2.37985521351e-22              1        51
        C(Duration, Sum)                   5.367071    0.0245738436636              1        51
        C(Weight, Sum)                    12.432445  3.99943118767e-05              2        51
        C(Duration, Sum):C(Weight, Sum)    0.176002      0.83912310946              2        51

        >>> res_poi = Poisson.from_formula("Days ~ C(Weight) * C(Duration)",                                            data).fit(cov_type='HC0')
        >>> wt = res_poi.wald_test_terms(skip_single=False,                                          combine_terms=['Duration', 'Weight'])
        >>> print(wt)
                                    chi2             P>chi2  df constraint
        Intercept              15.695625  7.43960374424e-05              1
        C(Weight)              16.132616  0.000313940174705              2
        C(Duration)             1.009147     0.315107378931              1
        C(Weight):C(Duration)   0.216694     0.897315972824              2
        Duration               11.187849     0.010752286833              3
        Weight                 30.263368  4.32586407145e-06              4
        """
    from collections import defaultdict
    result = self
    if extra_constraints is None:
        extra_constraints = []
    if combine_terms is None:
        combine_terms = []
    design_info = getattr(result.model.data, 'design_info', None)
    if design_info is None and extra_constraints is None:
        raise ValueError('no constraints, nothing to do')
    identity = np.eye(len(result.params))
    constraints = []
    combined = defaultdict(list)
    if design_info is not None:
        for term in design_info.terms:
            cols = design_info.slice(term)
            name = term.name()
            constraint_matrix = identity[cols]
            for cname in combine_terms:
                if cname in name:
                    combined[cname].append(constraint_matrix)
            k_constraint = constraint_matrix.shape[0]
            if skip_single:
                if k_constraint == 1:
                    continue
            constraints.append((name, constraint_matrix))
        combined_constraints = []
        for cname in combine_terms:
            combined_constraints.append((cname, np.vstack(combined[cname])))
    else:
        for col, name in enumerate(result.model.exog_names):
            constraint_matrix = np.atleast_2d(identity[col])
            for cname in combine_terms:
                if cname in name:
                    combined[cname].append(constraint_matrix)
            if skip_single:
                continue
            constraints.append((name, constraint_matrix))
        combined_constraints = []
        for cname in combine_terms:
            combined_constraints.append((cname, np.vstack(combined[cname])))
    use_t = result.use_t
    distribution = ['chi2', 'F'][use_t]
    res_wald = []
    index = []
    for name, constraint in constraints + combined_constraints + extra_constraints:
        wt = result.wald_test(constraint, scalar=scalar)
        row = [wt.statistic, wt.pvalue, constraint.shape[0]]
        if use_t:
            row.append(wt.df_denom)
        res_wald.append(row)
        index.append(name)
    col_names = ['statistic', 'pvalue', 'df_constraint']
    if use_t:
        col_names.append('df_denom')
    from pandas import DataFrame
    table = DataFrame(res_wald, index=index, columns=col_names)
    res = WaldTestResults(None, distribution, None, table=table)
    res.temp = constraints + combined_constraints + extra_constraints
    return res