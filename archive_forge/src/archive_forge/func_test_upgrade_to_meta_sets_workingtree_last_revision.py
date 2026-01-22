import os
import sys
from ... import (branch, controldir, errors, repository, upgrade, urlutils,
from ...bzr import bzrdir
from ...bzr.tests import test_bundle
from ...osutils import getcwd
from ...tests import TestCaseWithTransport
from ...tests.test_sftp_transport import TestCaseWithSFTPServer
from .branch import BzrBranchFormat4
from .bzrdir import BzrDirFormat5, BzrDirFormat6
def test_upgrade_to_meta_sets_workingtree_last_revision(self):
    self.build_tree_contents(_upgrade_dir_template)
    upgrade.upgrade('.', bzrdir.BzrDirMetaFormat1())
    tree = workingtree.WorkingTree.open('.')
    self.addCleanup(tree.lock_read().unlock)
    self.assertEqual([tree.branch._revision_history()[-1]], tree.get_parent_ids())