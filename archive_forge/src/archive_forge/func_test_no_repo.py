import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_no_repo(self):
    dir = controldir.ControlDir.create(self.get_url())
    error = errors.NoRepositoryPresent(dir)
    self.assertNotEqual(-1, str(error).find(dir.transport.clone('..').base))
    self.assertEqual(-1, str(error).find(dir.transport.base))