from collections import defaultdict
import re
from .pyutil import memoize
from .periodic import symbols
def formula_to_unicode(formula, prefixes=None, infixes=None, **kwargs):
    """Convert formula string to unicode string representation

    Parameters
    ----------
    formula : str
        Chemical formula, e.g. 'H2O', 'Fe+3', 'Cl-'
    prefixes : dict
        Prefix transofmrations, default: greek letters and .
    infixes : dict
        Infix transofmrations, default: .
    suffixes : tuple of strings
        Suffixes to keep, e.g. ('(g)', '(s)')

    Examples
    --------
    >>> formula_to_unicode('NH4+') == u'NH₄⁺'
    True
    >>> formula_to_unicode('Fe(CN)6+2') == u'Fe(CN)₆²⁺'
    True
    >>> formula_to_unicode('Fe(CN)6+2(aq)') == u'Fe(CN)₆²⁺(aq)'
    True
    >>> formula_to_unicode('.NHO-(aq)') == u'⋅NHO⁻(aq)'
    True
    >>> formula_to_unicode('alpha-FeOOH(s)') == u'α-FeOOH(s)'
    True

    """
    if prefixes is None:
        prefixes = _unicode_mapping
    if infixes is None:
        infixes = _unicode_infix_mapping
    return _formula_to_format(lambda x: ''.join((_unicode_sub[str(_)] for _ in x)), lambda x: ''.join((_unicode_sup[str(_)] for _ in x)), formula, prefixes, infixes, **kwargs)