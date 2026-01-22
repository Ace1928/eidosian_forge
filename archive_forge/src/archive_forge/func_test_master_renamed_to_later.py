from breezy import multiwalker, revision
from breezy import tree as _mod_tree
from breezy.tests import TestCaseWithTransport
def test_master_renamed_to_later(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/a', 'tree/b', 'tree/d'])
    tree.add(['a', 'b', 'd'], ids=[b'a-id', b'b-id', b'd-id'])
    tree.commit('first', rev_id=b'first-rev-id')
    tree.rename_one('b', 'e')
    basis_tree, root_id = self.lock_and_get_basis_and_root_id(tree)
    walker = multiwalker.MultiWalker(tree, [basis_tree])
    iterator = walker.iter_all()
    self.assertWalkerNext('', root_id, True, [''], iterator)
    self.assertWalkerNext('a', b'a-id', True, ['a'], iterator)
    self.assertWalkerNext('d', b'd-id', True, ['d'], iterator)
    self.assertWalkerNext('e', b'b-id', True, ['b'], iterator)
    self.assertRaises(StopIteration, next, iterator)