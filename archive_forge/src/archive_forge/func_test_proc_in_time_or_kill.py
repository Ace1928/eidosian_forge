import os
import platform
import subprocess
import errno
import time
import sys
import unittest
import tempfile
def test_proc_in_time_or_kill(self):
    ret_code, response = proc_in_time_or_kill([sys.executable, '-c', 'while True: pass'], time_out=1)
    self.assertIn('rocess timed out', ret_code)
    self.assertIn('successfully terminated', ret_code)