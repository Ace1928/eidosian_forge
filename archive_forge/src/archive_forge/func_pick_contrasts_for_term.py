from __future__ import print_function
from patsy.util import no_pickling
def pick_contrasts_for_term(term, numeric_factors, used_subterms):
    categorical_factors = [f for f in term.factors if f not in numeric_factors]
    subterms = []
    for subset in _subsets_sorted(categorical_factors):
        subterm = _Subterm([_ExpandedFactor(False, f) for f in subset])
        if subterm not in used_subterms:
            subterms.append(subterm)
    used_subterms.update(subterms)
    _simplify_subterms(subterms)
    factor_codings = []
    for subterm in subterms:
        factor_coding = {}
        for expanded in subterm.efactors:
            factor_coding[expanded.factor] = expanded.includes_intercept
        factor_codings.append(factor_coding)
    return factor_codings