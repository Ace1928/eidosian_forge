from collections import defaultdict
import re
from .pyutil import memoize
from .periodic import symbols
def formula_to_latex(formula, prefixes=None, infixes=None, **kwargs):
    """Convert formula string to latex representation

    Parameters
    ----------
    formula: str
        Chemical formula, e.g. 'H2O', 'Fe+3', 'Cl-'
    prefixes: dict
        Prefix transformations, default: greek letters and .
    infixes: dict
        Infix transformations, default: .
    suffixes: iterable of str
        What suffixes not to interpret, default: (s), (l), (g), (aq)

    Examples
    --------
    >>> formula_to_latex('NH4+')
    'NH_{4}^{+}'
    >>> formula_to_latex('Fe(CN)6+2')
    'Fe(CN)_{6}^{2+}'
    >>> formula_to_latex('Fe(CN)6+2(aq)')
    'Fe(CN)_{6}^{2+}(aq)'
    >>> formula_to_latex('.NHO-(aq)')
    '^\\\\bullet NHO^{-}(aq)'
    >>> formula_to_latex('alpha-FeOOH(s)')
    '\\\\alpha-FeOOH(s)'

    """
    if prefixes is None:
        prefixes = _latex_mapping
    if infixes is None:
        infixes = _latex_infix_mapping
    return _formula_to_format(lambda x: '_{%s}' % x, lambda x: '^{%s}' % x, re.sub('([{}])', '\\\\\\1', formula) if re.search('[{}]', formula) else formula, prefixes, infixes, **kwargs)