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
def test_branch_stacked_branch_stacked(self):
    """Asking to stack on a stacked branch does work"""
    trunk_tree = self.make_branch_and_tree('target', format='1.9')
    trunk_revid = trunk_tree.commit('mainline')
    branch_tree = self.make_branch_and_tree('branch', format='1.9')
    branch_tree.branch.set_stacked_on_url(trunk_tree.branch.base)
    work_tree = trunk_tree.branch.controldir.sprout('local').open_workingtree()
    branch_revid = work_tree.commit('moar work plz')
    work_tree.branch.push(branch_tree.branch)
    out, err = self.run_bzr(['branch', 'branch', '--stacked', 'branch2'])
    self.assertEqual('', out)
    self.assertEqual('Created new stacked branch referring to %s.\n' % branch_tree.branch.base, err)
    self.assertEqual(branch_tree.branch.base, branch.Branch.open('branch2').get_stacked_on_url())
    branch2_tree = WorkingTree.open('branch2')
    branch2_revid = work_tree.commit('work on second stacked branch')
    work_tree.branch.push(branch2_tree.branch)
    self.assertRevisionInRepository('branch2', branch2_revid)
    self.assertRevisionsInBranchRepository([trunk_revid, branch_revid, branch2_revid], 'branch2')