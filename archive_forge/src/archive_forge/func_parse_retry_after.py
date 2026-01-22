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
def parse_retry_after(self, retry_after):
    if re.match('^\\s*[0-9]+\\s*$', retry_after):
        seconds = int(retry_after)
    else:
        retry_date_tuple = email.utils.parsedate_tz(retry_after)
        if retry_date_tuple is None:
            raise InvalidHeader('Invalid Retry-After header: %s' % retry_after)
        if retry_date_tuple[9] is None:
            retry_date_tuple = retry_date_tuple[:9] + (0,) + retry_date_tuple[10:]
        retry_date = email.utils.mktime_tz(retry_date_tuple)
        seconds = retry_date - time.time()
    if seconds < 0:
        seconds = 0
    return seconds