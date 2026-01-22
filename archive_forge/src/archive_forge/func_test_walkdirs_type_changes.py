import os
from breezy.tests.features import SymlinkFeature
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_walkdirs_type_changes(self):
    """Walkdir shows the actual kinds on disk and the recorded kinds."""
    self.requireFeature(SymlinkFeature(self.test_dir))
    tree = self.make_branch_and_tree('.')
    paths = ['file1', 'file2', 'dir1/', 'dir2/']
    self.build_tree(paths)
    tree.add(paths)
    tt = tree.transform()
    root_transaction_id = tt.trans_id_tree_path('')
    tt.new_symlink('link1', root_transaction_id, 'link-target', b'link1')
    tt.new_symlink('link2', root_transaction_id, 'link-target', b'link2')
    tt.apply()
    tree.controldir.root_transport.delete_tree('dir1')
    tree.controldir.root_transport.delete_tree('dir2')
    tree.controldir.root_transport.delete('file1')
    tree.controldir.root_transport.delete('file2')
    tree.controldir.root_transport.delete('link1')
    tree.controldir.root_transport.delete('link2')
    changed_paths = ['dir1', 'file1/', 'link1', 'link2/']
    self.build_tree(changed_paths)
    os.symlink('target', 'dir2')
    os.symlink('target', 'file2')
    dir1_stat = os.lstat('dir1')
    dir2_stat = os.lstat('dir2')
    file1_stat = os.lstat('file1')
    file2_stat = os.lstat('file2')
    link1_stat = os.lstat('link1')
    link2_stat = os.lstat('link2')
    expected_dirblocks = [('', [('dir1', 'dir1', 'file', dir1_stat, 'directory' if tree.has_versioned_directories() else None), ('dir2', 'dir2', 'symlink', dir2_stat, 'directory' if tree.has_versioned_directories() else None), ('file1', 'file1', 'directory', file1_stat, 'file'), ('file2', 'file2', 'symlink', file2_stat, 'file'), ('link1', 'link1', 'file', link1_stat, 'symlink'), ('link2', 'link2', 'directory', link2_stat, 'symlink')])]
    if tree.has_versioned_directories():
        expected_dirblocks.extend([('dir1', []), ('dir2', [])])
    expected_dirblocks.extend([('file1', []), ('link2', [])])
    with tree.lock_read():
        result = list(tree.walkdirs())
    for pos, item in enumerate(expected_dirblocks):
        self.assertEqual(item, result[pos])
    self.assertEqual(len(expected_dirblocks), len(result))