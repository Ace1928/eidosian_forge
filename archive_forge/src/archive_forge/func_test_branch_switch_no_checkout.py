import os
from breezy import branch, controldir, errors
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import bzrdir
from breezy.bzr.knitrepo import RepositoryFormatKnit1
from breezy.tests import fixtures, test_server
from breezy.tests.blackbox import test_switch
from breezy.tests.features import HardlinkFeature
from breezy.tests.script import run_script
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
from breezy.urlutils import local_path_to_url, strip_trailing_slash
from breezy.workingtree import WorkingTree
def test_branch_switch_no_checkout(self):
    self.example_branch('a')
    tree = self.make_branch_and_tree('current')
    c1 = tree.commit('some diverged change')
    self.run_bzr_error(['Cannot switch a branch, only a checkout'], 'branch --switch ../a ../b', working_dir='current')
    a = branch.Branch.open('a')
    b = branch.Branch.open('b')
    self.assertEqual(a.last_revision(), b.last_revision())
    work = branch.Branch.open('current')
    self.assertEqual(work.last_revision(), c1)