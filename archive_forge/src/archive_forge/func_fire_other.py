import os
import signal
from testtools.helpers import try_import
from testtools import skipIf
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
def fire_other():
    """Fire Deferred from the last spin while waiting for this one."""
    deferred1.callback(object())
    return deferred2