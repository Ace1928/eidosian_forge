import os
import sys
import subprocess
import time
from pecan.compat import urlopen, URLError
from pecan.tests import PecanTestCase
import unittest
def test_project_pecan_serve_command(self):
    proc = subprocess.Popen([os.path.join(self.bin, 'pecan'), 'serve', 'testing123/config.py'])
    try:
        self.poll(proc)
        retries = 30
        while True:
            retries -= 1
            if retries < 0:
                raise RuntimeError('The HTTP server has not replied within 3 seconds.')
            try:
                resp = urlopen('http://localhost:8080/')
                assert resp.getcode()
                assert len(resp.read().decode())
            except URLError:
                pass
            else:
                break
            time.sleep(0.1)
    finally:
        proc.terminate()