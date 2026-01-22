import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_transport_not_possible(self):
    error = errors.TransportNotPossible('readonly', 'original error')
    self.assertEqualDiff('Transport operation not possible: readonly original error', str(error))