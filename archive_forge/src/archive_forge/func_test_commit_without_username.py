import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_commit_without_username(self):
    """Ensure commit error if username is not set.
        """
    self.run_bzr(['init', 'foo'])
    with open('foo/foo.txt', 'w') as f:
        f.write('hello')
    self.run_bzr(['add'], working_dir='foo')
    override_whoami(self)
    self.run_bzr_error(['Unable to determine your name'], ['commit', '-m', 'initial'], working_dir='foo')