from __future__ import absolute_import
import email
import logging
import re
import time
import warnings
from collections import namedtuple
from itertools import takewhile
from ..exceptions import (
from ..packages import six
def get_retry_after(self, response):
    """Get the value of Retry-After in seconds."""
    retry_after = response.getheader('Retry-After')
    if retry_after is None:
        return None
    return self.parse_retry_after(retry_after)