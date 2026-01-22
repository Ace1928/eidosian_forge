import os
from breezy import tests
from breezy.mutabletree import MutableTree
from breezy.osutils import supports_symlinks
from breezy.tests.per_tree import TestCaseWithTree
def test_walkdir_subtree(self):
    tree = self.get_tree_with_subdirs_and_all_supported_content_types(supports_symlinks(self.test_dir))
    result = []
    with tree.lock_read():
        expected_dirblocks = self.get_all_subdirs_expected(tree, supports_symlinks(self.test_dir))[1:]
        for dirinfo, block in tree.walkdirs('1top-dir'):
            newblock = []
            for row in block:
                if row[4] is not None:
                    newblock.append(row[0:3] + (None,) + row[4:])
                else:
                    newblock.append(row)
            result.append((dirinfo, newblock))
    for pos, item in enumerate(expected_dirblocks):
        self.assertEqual(item, result[pos])
    self.assertEqual(len(expected_dirblocks), len(result))