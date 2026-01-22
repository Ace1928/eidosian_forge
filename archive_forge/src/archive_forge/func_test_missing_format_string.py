import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_missing_format_string(self):
    e = ErrorWithNoFormat(param='randomvalue')
    self.assertStartsWith(str(e), 'Unprintable exception ErrorWithNoFormat')