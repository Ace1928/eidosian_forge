import shutil
from breezy import errors
from breezy.tests import TestNotApplicable, TestSkipped, features, per_tree
from breezy.tree import MissingNestedTree
def test_tree_with_merged_utf8(self):
    wt = self.make_branch_and_tree('.')
    self._create_tree_with_utf8(wt)
    tree2 = wt.controldir.sprout('tree2').open_workingtree()
    self.build_tree(['tree2/ba€r/qu€x'])
    if wt.supports_setting_file_ids():
        tree2.add(['ba€r/qu€x'], ids=['qu€x-id'.encode()])
    else:
        tree2.add(['ba€r/qu€x'])
    if wt.branch.repository._format.supports_setting_revision_ids:
        tree2.commit('to mérge', rev_id='rév-2'.encode())
    else:
        tree2.commit('to mérge')
    self.assertTrue(tree2.is_versioned('ba€r/qu€x'))
    wt.merge_from_branch(tree2.branch)
    self.assertTrue(wt.is_versioned('ba€r/qu€x'))
    if wt.branch.repository._format.supports_setting_revision_ids:
        wt.commit('mérge', rev_id='rév-3'.encode())
    else:
        wt.commit('mérge')
    tree = self.workingtree_to_test_tree(wt)
    revision_id_1 = 'rév-1'.encode()
    revision_id_2 = 'rév-2'.encode()
    root_id = b'TREE_ROOT'
    bar_id = 'ba€r-id'.encode()
    foo_id = 'fo€o-id'.encode()
    baz_id = 'ba€z-id'.encode()
    qux_id = 'qu€x-id'.encode()
    path_and_ids = [('', root_id, None, None), ('ba€r', bar_id, root_id, revision_id_1), ('fo€o', foo_id, root_id, revision_id_1), ('ba€r/ba€z', baz_id, bar_id, revision_id_1), ('ba€r/qu€x', qux_id, bar_id, revision_id_2)]
    with tree.lock_read():
        path_entries = list(tree.iter_entries_by_dir())
    for (epath, efid, eparent, erev), (path, ie) in zip(path_and_ids, path_entries):
        self.assertEqual(epath, path)
        self.assertIsInstance(path, str)
        self.assertIsInstance(ie.file_id, bytes)
        if wt.supports_setting_file_ids():
            self.assertEqual(efid, ie.file_id)
            self.assertEqual(eparent, ie.parent_id)
        if eparent is not None:
            self.assertIsInstance(ie.parent_id, bytes)
    self.assertEqual(len(path_and_ids), len(path_entries), '{!r} vs {!r}'.format([p for p, f, pf, r in path_and_ids], [p for p, e in path_entries]))
    get_revision_id = getattr(tree, 'get_revision_id', None)
    if get_revision_id is not None:
        self.assertIsInstance(get_revision_id(), bytes)
    last_revision = getattr(tree, 'last_revision', None)
    if last_revision is not None:
        self.assertIsInstance(last_revision(), bytes)