from __future__ import division
import sys as _sys
import datetime as _datetime
import uuid as _uuid
import traceback as _traceback
import os as _os
import logging as _logging
from syslog import (LOG_EMERG, LOG_ALERT, LOG_CRIT, LOG_ERR,
from ._journal import __version__, sendv, stream_fd
from ._reader import (_Reader, NOP, APPEND, INVALIDATE,
from . import id128 as _id128
def log_level(self, level):
    """Set maximum log `level` by setting matches for PRIORITY.
        """
    if 0 <= level <= 7:
        for i in range(level + 1):
            self.add_match(PRIORITY='%d' % i)
    else:
        raise ValueError('Log level must be 0 <= level <= 7')