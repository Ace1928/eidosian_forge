import os
import signal
import sys
import time
from breezy import debug, tests
def test_dash_derror(self):
    """With -Derror, tracebacks are shown even for user errors"""
    out, err = self.run_bzr('-Derror branch nonexistent-location', retcode=3)
    self.assertContainsRe(err, 'Traceback \\(most recent call last\\)')