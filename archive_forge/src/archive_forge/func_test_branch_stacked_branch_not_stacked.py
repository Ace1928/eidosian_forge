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
def test_branch_stacked_branch_not_stacked(self):
    """Branching a stacked branch is not stacked by default"""
    trunk_tree = self.make_branch_and_tree('target', format='1.9')
    trunk_tree.commit('mainline')
    branch_tree = self.make_branch_and_tree('branch', format='1.9')
    branch_tree.branch.set_stacked_on_url(trunk_tree.branch.base)
    work_tree = trunk_tree.branch.controldir.sprout('local').open_workingtree()
    work_tree.commit('moar work plz')
    work_tree.branch.push(branch_tree.branch)
    out, err = self.run_bzr(['branch', 'branch', 'newbranch'])
    self.assertEqual('', out)
    self.assertEqual('Branched 2 revisions.\n', err)
    self.assertRaises(errors.NotStacked, controldir.ControlDir.open('newbranch').open_branch().get_stacked_on_url)