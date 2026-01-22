import sys
import encodings
import encodings.aliases
import re
import _collections_abc
from builtins import str as _builtin_str
import functools
def currency(val, symbol=True, grouping=False, international=False):
    """Formats val according to the currency settings
    in the current locale."""
    conv = localeconv()
    digits = conv[international and 'int_frac_digits' or 'frac_digits']
    if digits == 127:
        raise ValueError("Currency formatting is not possible using the 'C' locale.")
    s = _localize(f'{abs(val):.{digits}f}', grouping, monetary=True)
    s = '<' + s + '>'
    if symbol:
        smb = conv[international and 'int_curr_symbol' or 'currency_symbol']
        precedes = conv[val < 0 and 'n_cs_precedes' or 'p_cs_precedes']
        separated = conv[val < 0 and 'n_sep_by_space' or 'p_sep_by_space']
        if precedes:
            s = smb + (separated and ' ' or '') + s
        else:
            if international and smb[-1] == ' ':
                smb = smb[:-1]
            s = s + (separated and ' ' or '') + smb
    sign_pos = conv[val < 0 and 'n_sign_posn' or 'p_sign_posn']
    sign = conv[val < 0 and 'negative_sign' or 'positive_sign']
    if sign_pos == 0:
        s = '(' + s + ')'
    elif sign_pos == 1:
        s = sign + s
    elif sign_pos == 2:
        s = s + sign
    elif sign_pos == 3:
        s = s.replace('<', sign)
    elif sign_pos == 4:
        s = s.replace('>', sign)
    else:
        s = sign + s
    return s.replace('<', '').replace('>', '')