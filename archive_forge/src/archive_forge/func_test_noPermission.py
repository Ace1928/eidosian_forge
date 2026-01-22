from __future__ import annotations
import contextlib
import errno
import os
import stat
import time
from unittest import skipIf
from twisted.python import logfile, runtime
from twisted.trial.unittest import TestCase
def test_noPermission(self) -> None:
    """
        Check it keeps working when permission on dir changes.
        """
    log = logfile.LogFile(self.name, self.dir)
    self.addCleanup(log.close)
    log.write('abc')
    os.chmod(self.dir, 365)
    try:
        f = open(os.path.join(self.dir, 'xxx'), 'w')
    except OSError:
        pass
    else:
        f.close()
        return
    log.rotate()
    log.write('def')
    log.flush()
    f = log._file
    self.assertEqual(f.tell(), 6)
    f.seek(0, 0)
    self.assertEqual(f.read(), b'abcdef')