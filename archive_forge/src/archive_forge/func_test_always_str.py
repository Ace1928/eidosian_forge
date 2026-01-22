import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_always_str(self):
    e = PassThroughError('µ', 'bar')
    self.assertIsInstance(e.__str__(), str)
    s = str(e)
    self.assertEqual('Pass through µ and bar', s)