import collections
import operator
import re
import warnings
import abc
from debtcollector import removals
import netaddr
import rfc3986
def quote_trailing_and_leading_space(self, str_val):
    if not isinstance(str_val, str):
        warnings.warn("converting '%s' to a string" % str_val)
        str_val = str(str_val)
    if str_val.strip() != str_val:
        return '"%s"' % str_val
    return str_val