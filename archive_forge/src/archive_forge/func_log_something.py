import os
import signal
from testtools.helpers import try_import
from testtools import skipIf
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
@_spinner.not_reentrant
def log_something():
    calls.append(None)
    if len(calls) < 5:
        log_something()