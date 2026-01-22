import os
import signal
from testtools.helpers import try_import
from testtools import skipIf
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
def make_deferred_but_dont_handle():
    try:
        1 / 0
    except ZeroDivisionError:
        f = Failure()
        failures.append(f)
        defer.fail(f)