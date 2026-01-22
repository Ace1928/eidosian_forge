from breezy import errors, tests
from breezy import transport as _mod_transport
from breezy.tests import per_workingtree
def make_branch_with_merged_deletions(self, relpath='tree'):
    tree = self.make_branch_and_tree(relpath)
    files = ['a', 'b/', 'b/c']
    self.build_tree(files, line_endings='binary', transport=tree.controldir.root_transport)
    tree.add(files)
    base_revid = tree.commit('a, b and b/c')
    tree2 = tree.controldir.sprout(relpath + '2').open_workingtree()
    tree.remove('a', keep_files=False)
    this_revid = tree.commit('remove a')
    tree2.remove('b/c', keep_files=False)
    tree2.remove('b', keep_files=False)
    other_revid = tree2.commit('remove b/c')
    tree.merge_from_branch(tree2.branch)
    return (tree, [base_revid, this_revid, other_revid])