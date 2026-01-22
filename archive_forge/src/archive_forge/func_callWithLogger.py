import sys
import time
import warnings
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, BinaryIO, Dict, Optional, cast
from zope.interface import Interface
from twisted.logger import (
from twisted.logger._global import LogBeginner
from twisted.logger._legacy import publishToNewObserver as _publishNew
from twisted.python import context, failure, reflect, util
from twisted.python.threadable import synchronize
def callWithLogger(logger, func, *args, **kw):
    """
    Utility method which wraps a function in a try:/except:, logs a failure if
    one occurs, and uses the system's logPrefix.
    """
    try:
        lp = logger.logPrefix()
    except KeyboardInterrupt:
        raise
    except BaseException:
        lp = '(buggy logPrefix method)'
        err(system=lp)
    try:
        return callWithContext({'system': lp}, func, *args, **kw)
    except KeyboardInterrupt:
        raise
    except BaseException:
        err(system=lp)