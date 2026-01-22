import os
import sys
from breezy import (branch, debug, osutils, tests, uncommit, urlutils,
from breezy.bzr import remote
from breezy.directory_service import directories
from breezy.tests import fixtures, script
def test_pull_from_directory_service(self):
    source = self.make_branch_and_tree('source')
    source.commit('commit 1')
    target = source.controldir.sprout('target').open_workingtree()
    source_last = source.commit('commit 2')

    class FooService:
        """A directory service that always returns source"""

        def look_up(self, name, url, purpose=None):
            return 'source'
    directories.register('foo:', FooService, 'Testing directory service')
    self.addCleanup(directories.remove, 'foo:')
    self.run_bzr('pull foo:bar -d target')
    self.assertEqual(source_last, target.last_revision())