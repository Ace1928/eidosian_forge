import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def test_push_use_existing(self):
    """'brz push --use-existing-dir' can push into an existing dir.

        By default, 'brz push' will not use an existing, non-versioned dir.
        """
    tree = self.create_simple_tree()
    self.build_tree(['target/'])
    self.run_bzr_error(['Target directory ../target already exists', 'Supply --use-existing-dir'], 'push ../target', working_dir='tree')
    self.run_bzr('push --use-existing-dir ../target', working_dir='tree')
    new_tree = workingtree.WorkingTree.open('target')
    self.assertEqual(tree.last_revision(), new_tree.last_revision())
    self.assertPathExists('target/a')