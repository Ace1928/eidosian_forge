import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_lock_active(self):
    error = errors.LockActive('lock description')
    self.assertEqualDiff("The lock for 'lock description' is in use and cannot be broken.", str(error))