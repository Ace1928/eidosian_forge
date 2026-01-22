import itertools
from collections import OrderedDict
import numpy as np
def find_output_str(subscripts):
    """
    Find the output string for the inputs ``subscripts`` under canonical einstein summation rules. That is, repeated indices are summed over by default.

    Examples
    --------
    >>> oe.parser.find_output_str("ab,bc")
    'ac'

    >>> oe.parser.find_output_str("a,b")
    'ab'

    >>> oe.parser.find_output_str("a,a,b,b")
    ''
    """
    tmp_subscripts = subscripts.replace(',', '')
    return ''.join((s for s in sorted(set(tmp_subscripts)) if tmp_subscripts.count(s) == 1))