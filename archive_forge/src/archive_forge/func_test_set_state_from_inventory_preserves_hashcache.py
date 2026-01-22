import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_set_state_from_inventory_preserves_hashcache(self):
    tree = self.make_branch_and_tree('.')
    with tree.lock_write():
        foo_contents = b'contents of foo'
        self.build_tree_contents([('foo', foo_contents)])
        tree.add('foo', ids=b'foo-id')
        foo_stat = os.stat('foo')
        foo_packed = dirstate.pack_stat(foo_stat)
        foo_sha = osutils.sha_string(foo_contents)
        foo_size = len(foo_contents)
        self.assertEqual(((b'', b'foo', b'foo-id'), [(b'f', b'', 0, False, dirstate.DirState.NULLSTAT)]), tree._dirstate._get_entry(0, b'foo-id'))
        tree._dirstate.update_minimal((b'', b'foo', b'foo-id'), b'f', False, foo_sha, foo_packed, foo_size, b'foo')
        self.assertEqual(((b'', b'foo', b'foo-id'), [(b'f', foo_sha, foo_size, False, foo_packed)]), tree._dirstate._get_entry(0, b'foo-id'))
        inv = tree._get_root_inventory()
        self.assertTrue(inv.has_id(b'foo-id'))
        self.assertTrue(inv.has_filename('foo'))
        inv.add_path('bar', 'file', b'bar-id')
        tree._dirstate._validate()
        tree._dirstate.set_state_from_inventory(inv)
        tree._dirstate._validate()
    with tree.lock_read():
        state = tree._dirstate
        state._validate()
        foo_tuple = state._get_entry(0, path_utf8=b'foo')
        self.assertEqual(((b'', b'foo', b'foo-id'), [(b'f', foo_sha, len(foo_contents), False, dirstate.pack_stat(foo_stat))]), foo_tuple)