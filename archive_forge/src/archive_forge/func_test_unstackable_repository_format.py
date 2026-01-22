import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_unstackable_repository_format(self):
    format = 'foo'
    url = '/foo'
    error = errors.UnstackableRepositoryFormat(format, url)
    self.assertEqualDiff("The repository '/foo'(foo) is not a stackable format. You will need to upgrade the repository to permit branch stacking.", str(error))