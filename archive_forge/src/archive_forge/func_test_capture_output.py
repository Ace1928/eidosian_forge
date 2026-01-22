import sys
from io import StringIO
import unittest
from IPython.utils.io import Tee, capture_output
def test_capture_output(self):
    """capture_output() context works"""
    with capture_output() as io:
        print('hi, stdout')
        print('hi, stderr', file=sys.stderr)
    self.assertEqual(io.stdout, 'hi, stdout\n')
    self.assertEqual(io.stderr, 'hi, stderr\n')