import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_duplicate_help_prefix(self):
    error = errors.DuplicateHelpPrefix('foo')
    self.assertEqualDiff('The prefix foo is in the help search path twice.', str(error))