import logging
import sys
from typing import Union
from rq.defaults import DEFAULT_LOGGING_DATE_FORMAT, DEFAULT_LOGGING_FORMAT
@property
def is_tty(self):
    isatty = getattr(self.stream, 'isatty', None)
    return isatty and isatty()