from collections import defaultdict
import re
from .pyutil import memoize
from .periodic import symbols
def formula_to_html(formula, prefixes=None, infixes=None, **kwargs):
    """Convert formula string to html string representation

    Parameters
    ----------
    formula : str
        Chemical formula, e.g. 'H2O', 'Fe+3', 'Cl-'
    prefixes : dict
        Prefix transformations, default: greek letters and .
    infixes : dict
        Infix transformations, default: .
    suffixes : tuple of strings
        Suffixes to keep, e.g. ('(g)', '(s)')

    Examples
    --------
    >>> formula_to_html('NH4+')
    'NH<sub>4</sub><sup>+</sup>'
    >>> formula_to_html('Fe(CN)6+2')
    'Fe(CN)<sub>6</sub><sup>2+</sup>'
    >>> formula_to_html('Fe(CN)6+2(aq)')
    'Fe(CN)<sub>6</sub><sup>2+</sup>(aq)'
    >>> formula_to_html('.NHO-(aq)')
    '&sdot;NHO<sup>-</sup>(aq)'
    >>> formula_to_html('alpha-FeOOH(s)')
    '&alpha;-FeOOH(s)'

    """
    if prefixes is None:
        prefixes = _html_mapping
    if infixes is None:
        infixes = _html_infix_mapping
    return _formula_to_format(lambda x: '<sub>%s</sub>' % x, lambda x: '<sup>%s</sup>' % x, formula, prefixes, infixes, **kwargs)