import collections.abc
import math
import re
import unicodedata
import urllib
from oslo_utils._i18n import _
from oslo_utils import encodeutils
def is_valid_boolstr(value):
    """Check if the provided string is a valid bool string or not.

    :param value: value to verify
    :type value: string
    :returns: true if value is boolean string, false otherwise

    .. versionadded:: 3.17
    """
    boolstrs = TRUE_STRINGS + FALSE_STRINGS
    return str(value).lower() in boolstrs