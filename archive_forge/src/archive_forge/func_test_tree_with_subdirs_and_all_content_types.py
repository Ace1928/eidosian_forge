import shutil
from breezy import errors
from breezy.tests import TestNotApplicable, TestSkipped, features, per_tree
from breezy.tree import MissingNestedTree
def test_tree_with_subdirs_and_all_content_types(self):
    self.requireFeature(features.SymlinkFeature(self.test_dir))
    tree = self.get_tree_with_subdirs_and_all_content_types()
    tree.lock_read()
    self.addCleanup(tree.unlock)
    self.assertEqual([], tree.get_parent_ids())
    self.assertEqual([], tree.conflicts())
    self.assertEqual([], list(tree.unknowns()))
    try:
        all_file_ids = set(tree.all_file_ids())
        tree_root = tree.path2id('')
    except AttributeError:
        all_file_ids = None
    if tree.has_versioned_directories():
        if all_file_ids is not None:
            self.assertEqual({tree.path2id(p) for p in ['', '0file', '1top-dir', '1top-dir/1dir-in-1topdir', '1top-dir/0file-in-1topdir', 'symlink', '2utfሴfile']}, set(tree.all_file_ids()))
        self.assertEqual([('', 'directory'), ('0file', 'file'), ('1top-dir', 'directory'), ('2utfሴfile', 'file'), ('symlink', 'symlink'), ('1top-dir/0file-in-1topdir', 'file'), ('1top-dir/1dir-in-1topdir', 'directory')], [(path, node.kind) for path, node in tree.iter_entries_by_dir()])
    else:
        if all_file_ids is not None:
            self.assertEqual({tree.path2id(p) for p in ['', '0file', '1top-dir', '1top-dir/0file-in-1topdir', 'symlink', '2utfሴfile']}, set(tree.all_file_ids()))
        self.assertEqual([('', 'directory'), ('0file', 'file'), ('1top-dir', 'directory'), ('2utfሴfile', 'file'), ('symlink', 'symlink'), ('1top-dir/0file-in-1topdir', 'file')], [(path, node.kind) for path, node in tree.iter_entries_by_dir()])