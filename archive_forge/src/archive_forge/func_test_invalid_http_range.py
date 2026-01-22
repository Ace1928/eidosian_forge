import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_invalid_http_range(self):
    error = errors.InvalidHttpRange('path', 'Content-Range: potatoes 0-00/o0oo0', 'bad range')
    self.assertEqual("Invalid http range 'Content-Range: potatoes 0-00/o0oo0' for path: bad range", str(error))