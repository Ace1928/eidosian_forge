from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
def test_create_open_reference(self):
    bzrdirformat = bzrdir.BzrDirMetaFormat1()
    t = self.get_transport()
    t.mkdir('repo')
    dir = bzrdirformat.initialize(self.get_url('repo'))
    dir.create_repository()
    target_branch = dir.create_branch()
    t.mkdir('branch')
    branch_dir = bzrdirformat.initialize(self.get_url('branch'))
    made_branch = _mod_bzrbranch.BranchReferenceFormat().initialize(branch_dir, target_branch=target_branch)
    self.assertEqual(made_branch.base, target_branch.base)
    opened_branch = branch_dir.open_branch()
    self.assertEqual(opened_branch.base, target_branch.base)