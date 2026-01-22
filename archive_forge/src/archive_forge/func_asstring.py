import collections
import os
import re
import sys
from datetime import timedelta
from pyudev._errors import (
from pyudev._util import (
def asstring(self, attribute):
    """
        Get the given ``attribute`` for the device as unicode string.

        :param attribute: the key for an attribute value
        :type attribute: unicode or byte string
        :returns: the value corresponding to ``attribute``, as unicode
        :rtype: unicode
        :raises KeyError: if no value found for ``attribute``
        :raises UnicodeDecodeError: if value is not convertible
        """
    return ensure_unicode_string(self._get(attribute))