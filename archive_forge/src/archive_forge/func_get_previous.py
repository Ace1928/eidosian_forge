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
def get_previous(self, skip=1):
    """Return the previous log entry.

        Equivalent to get_next(-skip).

        Optional `skip` value will return the -`skip`-th log entry.

        Entries will be processed with converters specified during Reader
        creation.

        Currently a standard dictionary of fields is returned, but in the
        future this might be changed to a different mapping type, so the
        calling code should not make assumptions about a specific type.
        """
    return self.get_next(-skip)