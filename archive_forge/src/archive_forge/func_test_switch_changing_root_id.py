import os
from breezy import branch, errors
from breezy import merge as _mod_merge
from breezy import switch, tests, workingtree
def test_switch_changing_root_id(self):
    tree = self._setup_tree()
    tree2 = self.make_branch_and_tree('tree-2')
    tree2.set_root_id(b'custom-root-id')
    self.build_tree(['tree-2/file-2'])
    tree2.add(['file-2'])
    tree2.commit('rev1b')
    checkout = tree.branch.create_checkout('checkout', lightweight=self.lightweight)
    switch.switch(checkout.controldir, tree2.branch)
    self.assertEqual(b'custom-root-id', tree2.path2id(''))