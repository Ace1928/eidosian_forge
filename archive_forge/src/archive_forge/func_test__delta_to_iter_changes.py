from .... import branch, errors, osutils, tests
from ....bzr import inventory
from .. import revision_store
from . import FastimportFeature
def test__delta_to_iter_changes(self):
    basis_inv = self.make_trivial_basis_inv()
    foo_entry = inventory.make_entry('file', 'foo2', b'bar-id', b'foo-id')
    link_entry = inventory.make_entry('symlink', 'link', b'TREE_ROOT', b'link-id')
    link_entry.symlink_target = 'link-target'
    inv_delta = [('foo', 'bar/foo2', b'foo-id', foo_entry), ('bar/baz', None, b'baz-id', None), (None, 'link', b'link-id', link_entry)]
    shim = revision_store._TreeShim(repo=None, basis_inv=basis_inv, inv_delta=inv_delta, content_provider=None)
    changes = list(shim._delta_to_iter_changes())
    expected = [(b'foo-id', ('foo', 'bar/foo2'), False, (True, True), (b'TREE_ROOT', b'bar-id'), ('foo', 'foo2'), ('file', 'file'), (False, False), False), (b'baz-id', ('bar/baz', None), True, (True, False), (b'bar-id', None), ('baz', None), ('file', None), (False, None), False), (b'link-id', (None, 'link'), True, (False, True), (None, b'TREE_ROOT'), (None, 'link'), (None, 'symlink'), (None, False), False)]
    self.assertEqual(expected, changes)