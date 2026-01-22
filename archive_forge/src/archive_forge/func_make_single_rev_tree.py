from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def make_single_rev_tree(self):
    builder = self.make_branch_builder('branch')
    revid = builder.build_snapshot(None, [('add', ('', None, 'directory', None)), ('add', ('file', None, 'file', b'initial content\n'))])
    b = builder.get_branch()
    tree = b.create_checkout('tree', lightweight=True)
    tree.lock_read()
    self.addCleanup(tree.unlock)
    return (tree, revid)