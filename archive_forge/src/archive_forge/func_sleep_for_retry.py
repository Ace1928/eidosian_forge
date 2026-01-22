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
def sleep_for_retry(self, response=None):
    retry_after = self.get_retry_after(response)
    if retry_after:
        time.sleep(retry_after)
        return True
    return False