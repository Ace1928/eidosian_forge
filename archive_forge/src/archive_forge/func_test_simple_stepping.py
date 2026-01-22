from breezy import multiwalker, revision
from breezy import tree as _mod_tree
from breezy.tests import TestCaseWithTransport
def test_simple_stepping(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/a', 'tree/b/', 'tree/b/c'])
    tree.add(['a', 'b', 'b/c'], ids=[b'a-id', b'b-id', b'c-id'])
    tree.commit('first', rev_id=b'first-rev-id')
    basis_tree, root_id = self.lock_and_get_basis_and_root_id(tree)
    walker = multiwalker.MultiWalker(tree, [basis_tree])
    iterator = walker.iter_all()
    self.assertWalkerNext('', root_id, True, [''], iterator)
    self.assertWalkerNext('a', b'a-id', True, ['a'], iterator)
    self.assertWalkerNext('b', b'b-id', True, ['b'], iterator)
    self.assertWalkerNext('b/c', b'c-id', True, ['b/c'], iterator)
    self.assertRaises(StopIteration, next, iterator)