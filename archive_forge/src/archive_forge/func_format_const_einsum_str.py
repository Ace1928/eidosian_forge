from collections import namedtuple
from decimal import Decimal
import numpy as np
from . import backends, blas, helpers, parser, paths, sharing
def format_const_einsum_str(einsum_str, constants):
    """Add brackets to the constant terms in ``einsum_str``. For example:

        >>> format_const_einsum_str('ab,bc,cd->ad', [0, 2])
        'bc,[ab,cd]->ad'

    No-op if there are no constants.
    """
    if not constants:
        return einsum_str
    if '->' in einsum_str:
        lhs, rhs = einsum_str.split('->')
        arrow = '->'
    else:
        lhs, rhs, arrow = (einsum_str, '', '')
    wrapped_terms = ['[{}]'.format(t) if i in constants else t for i, t in enumerate(lhs.split(','))]
    formatted_einsum_str = '{}{}{}'.format(','.join(wrapped_terms), arrow, rhs)
    formatted_einsum_str = formatted_einsum_str.replace('],[', ',')
    return formatted_einsum_str