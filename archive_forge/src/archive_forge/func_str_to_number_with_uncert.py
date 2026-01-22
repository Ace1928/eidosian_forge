from __future__ import division  # Many analytical derivatives depend on this
from builtins import str, next, map, zip, range, object
import math
from math import sqrt, log, isnan, isinf  # Optimization: no attribute look-up
import re
import sys
import copy
import warnings
import itertools
import inspect
import numbers
import collections
def str_to_number_with_uncert(representation):
    """
    Given a string that represents a number with uncertainty, returns the
    nominal value and the uncertainty.

    See the documentation for ufloat_fromstr() for a list of accepted
    formats.

    When no numerical error is given, an uncertainty of 1 on the last
    digit is implied.

    Raises ValueError if the string cannot be parsed.

    representation -- string with no leading or trailing spaces.
    """
    if representation.startswith('(') and representation.endswith(')'):
        representation = representation[1:-1]
    match = NUMBER_WITH_UNCERT_GLOBAL_EXP_RE_MATCH(representation)
    if match:
        exp_value_str = match.group('exp_value')
        try:
            exponent = nrmlze_superscript(exp_value_str)
        except ValueError:
            raise ValueError(cannot_parse_ufloat_msg_pat % representation)
        factor = 10.0 ** exponent
        representation = match.group('simple_num_with_uncert')
    else:
        factor = 1
    match = re.match(u'(.*)(?:\\+/-|±)(.*)', representation)
    if match:
        nom_value, uncert = match.groups()
        try:
            parsed_value = (to_float(nom_value) * factor, to_float(uncert) * factor)
        except ValueError:
            raise ValueError(cannot_parse_ufloat_msg_pat % representation)
    else:
        try:
            parsed_value = parse_error_in_parentheses(representation)
        except NotParenUncert:
            raise ValueError(cannot_parse_ufloat_msg_pat % representation)
    return parsed_value