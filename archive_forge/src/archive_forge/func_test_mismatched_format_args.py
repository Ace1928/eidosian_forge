import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_mismatched_format_args(self):
    e = ErrorWithBadFormat(not_thing='x')
    self.assertStartsWith(str(e), 'Unprintable exception ErrorWithBadFormat')