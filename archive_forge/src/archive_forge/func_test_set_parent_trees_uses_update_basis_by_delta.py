import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
def test_set_parent_trees_uses_update_basis_by_delta(self):
    builder = self.make_branch_builder('source')
    builder.start_series()
    self.addCleanup(builder.finish_series)
    builder.build_snapshot([], [('add', ('', b'root-id', 'directory', None)), ('add', ('a', b'a-id', 'file', b'content\n'))], revision_id=b'A')
    builder.build_snapshot([b'A'], [('modify', ('a', b'new content\nfor a\n')), ('add', ('b', b'b-id', 'file', b'b-content\n'))], revision_id=b'B')
    tree = self.make_workingtree('tree')
    source_branch = builder.get_branch()
    tree.branch.repository.fetch(source_branch.repository, b'B')
    tree.pull(source_branch, stop_revision=b'A')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    state = tree.current_dirstate()
    called = []
    orig_update = state.update_basis_by_delta

    def log_update_basis_by_delta(delta, new_revid):
        called.append(new_revid)
        return orig_update(delta, new_revid)
    state.update_basis_by_delta = log_update_basis_by_delta
    basis = tree.basis_tree()
    self.assertEqual(b'a-id', basis.path2id('a'))
    self.assertFalse(basis.is_versioned('b'))

    def fail_set_parent_trees(trees, ghosts):
        raise AssertionError('dirstate.set_parent_trees() was called')
    state.set_parent_trees = fail_set_parent_trees
    tree.pull(source_branch, stop_revision=b'B')
    self.assertEqual([b'B'], called)
    basis = tree.basis_tree()
    self.assertEqual(b'a-id', basis.path2id('a'))
    self.assertEqual(b'b-id', basis.path2id('b'))