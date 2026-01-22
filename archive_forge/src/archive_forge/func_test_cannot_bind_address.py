import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_cannot_bind_address(self):
    e = errors.CannotBindAddress('example.com', 22, socket.error(13, 'Permission denied'))
    self.assertContainsRe(str(e), 'Cannot bind address "example\\.com:22":.*Permission denied')