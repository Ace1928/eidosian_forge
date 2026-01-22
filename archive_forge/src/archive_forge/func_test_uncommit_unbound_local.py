from .. import errors, tests, uncommit
def test_uncommit_unbound_local(self):
    tree, history = self.make_linear_tree()
    self.assertRaises(errors.LocalRequiresBoundBranch, uncommit.uncommit, tree.branch, tree=tree, local=True)