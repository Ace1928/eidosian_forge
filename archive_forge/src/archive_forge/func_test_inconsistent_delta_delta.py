import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_inconsistent_delta_delta(self):
    error = errors.InconsistentDeltaDelta([], 'reason')
    self.assertEqualDiff('An inconsistent delta was supplied: []\nreason: reason', str(error))