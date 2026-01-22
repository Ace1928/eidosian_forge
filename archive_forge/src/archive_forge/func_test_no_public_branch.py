import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_no_public_branch(self):
    b = self.make_branch('.')
    error = errors.NoPublicBranch(b)
    url = urlutils.unescape_for_display(b.base, 'ascii')
    self.assertEqualDiff('There is no public branch set for "%s".' % url, str(error))