import os
import shutil
from io import StringIO
from .. import bisect
from ..controldir import ControlDir
from . import TestCaseWithTransport, TestSkipped
def testShowLog(self):
    """Test that the log can be shown."""
    sio = StringIO()
    bisect.BisectCurrent(self.tree.controldir).show_rev_log(outf=sio)