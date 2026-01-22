import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_lock_corrupt(self):
    error = errors.LockCorrupt('corruption info')
    self.assertEqualDiff("Lock is apparently held, but corrupted: corruption info\nUse 'brz break-lock' to clear it", str(error))