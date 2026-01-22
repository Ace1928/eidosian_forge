from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
def test_post_switch_hook(self):
    from .. import switch
    calls = []
    _mod_branch.Branch.hooks.install_named_hook('post_switch', calls.append, None)
    tree = self.make_branch_and_tree('branch-1')
    self.build_tree(['branch-1/file-1'])
    tree.add('file-1')
    tree.commit('rev1')
    to_branch = tree.controldir.sprout('branch-2').open_branch()
    self.build_tree(['branch-1/file-2'])
    tree.add('file-2')
    tree.remove('file-1')
    tree.commit('rev2')
    checkout = tree.branch.create_checkout('checkout')
    self.assertLength(0, calls)
    switch.switch(checkout.controldir, to_branch)
    self.assertLength(1, calls)
    params = calls[0]
    self.assertIsInstance(params, _mod_branch.SwitchHookParams)
    self.assertTrue(hasattr(params, 'to_branch'))
    self.assertTrue(hasattr(params, 'revision_id'))