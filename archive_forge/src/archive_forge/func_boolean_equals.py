import logging
import re
from enum import Enum
from string import Formatter
from typing import NamedTuple
from botocore import xform_name
from botocore.compat import IPV4_RE, quote, urlparse
from botocore.exceptions import EndpointResolutionError
from botocore.utils import (
def boolean_equals(self, value1, value2):
    """Evaluates two boolean values for equality.

        :type value1: bool
        :type value2: bool
        :rtype: bool
        """
    if not all((isinstance(val, bool) for val in (value1, value2))):
        msg = f'Both arguments must be bools, not {type(value1)} and {type(value2)}.'
        raise EndpointResolutionError(msg=msg)
    return value1 is value2