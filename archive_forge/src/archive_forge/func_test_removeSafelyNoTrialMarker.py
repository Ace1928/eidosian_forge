from __future__ import annotations
import locale
import os
import sys
from io import StringIO
from typing import Generator
from zope.interface import implementer
from hamcrest import assert_that, equal_to
from twisted.internet.base import DelayedCall
from twisted.internet.interfaces import IProcessTransport
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.trial import util
from twisted.trial.unittest import SynchronousTestCase
from twisted.trial.util import (
def test_removeSafelyNoTrialMarker(self) -> None:
    """
        If a path doesn't contain a node named C{"_trial_marker"}, that path is
        not removed by L{util._removeSafely} and a L{util._NoTrialMarker}
        exception is raised instead.
        """
    directory = self.mktemp().encode('utf-8')
    os.mkdir(directory)
    dirPath = filepath.FilePath(directory)
    self.assertRaises(util._NoTrialMarker, util._removeSafely, dirPath)