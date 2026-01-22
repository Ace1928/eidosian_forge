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
@classmethod
def with_args(cls, config=None):
    """Create a JournalHandler with a configuration dictionary

        This creates a JournalHandler instance, but accepts the parameters through
        a dictionary that can be specified as a positional argument. This is useful
        in contexts like logging.config.fileConfig, where the syntax does not allow
        for positional arguments.

        >>> JournalHandler.with_args({'SYSLOG_IDENTIFIER':'my-cool-app'})
        <...JournalHandler ...>
        """
    return cls(**config or {})