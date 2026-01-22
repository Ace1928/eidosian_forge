import os
import shutil
from io import StringIO
from .. import bisect
from ..controldir import ControlDir
from . import TestCaseWithTransport, TestSkipped
def testSwitchVersions(self):
    """Test switching versions."""
    current = bisect.BisectCurrent(self.tree.controldir)
    self.assertRevno(5)
    current.switch(4)
    self.assertRevno(4)