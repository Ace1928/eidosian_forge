import os
import shutil
from io import StringIO
from .. import bisect
from ..controldir import ControlDir
from . import TestCaseWithTransport, TestSkipped
def testSave(self):
    """Test saving the log."""
    bisect_log = bisect.BisectLog(self.tree.controldir)
    bisect_log._items = [(b'rev1', 'yes'), (b'rev2', 'no'), (b'rev3', 'yes')]
    bisect_log.save()
    with open(os.path.join('.bzr', bisect.BISECT_INFO_PATH), 'rb') as logfile:
        self.assertEqual(logfile.read(), b'rev1 yes\nrev2 no\nrev3 yes\n')