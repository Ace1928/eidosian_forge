import collections.abc
import math
import re
import unicodedata
import urllib
from oslo_utils._i18n import _
from oslo_utils import encodeutils
def is_int_like(val):
    """Check if a value looks like an integer with base 10.

    :param val: Value to verify
    :type val: string
    :returns: bool

    .. versionadded:: 1.1
    """
    try:
        return str(int(val)) == str(val)
    except (TypeError, ValueError):
        return False