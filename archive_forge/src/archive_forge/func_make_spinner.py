import os
import signal
from testtools.helpers import try_import
from testtools import skipIf
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
def make_spinner(self, reactor=None):
    if reactor is None:
        reactor = self.make_reactor()
    return _spinner.Spinner(reactor)