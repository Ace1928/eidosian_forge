import os
import shutil
from .... import bedding, config, errors, trace
from ....merge import Merger
from ....mutabletree import MutableTree
from ....tests import TestCaseWithTransport, TestSkipped
from .. import (post_build_tree_quilt, post_merge_quilt_cleanup,
from ..merge import tree_unapply_patches
from ..quilt import QuiltPatches
from . import quilt_feature
def test_diverged_patches(self):
    self.enable_hooks()
    tree_a = self.make_branch_and_tree('a')
    self.build_tree(['a/debian/', 'a/debian/patches/', 'a/debian/source/', 'a/.pc/'])
    self.build_tree_contents([('a/.pc/.quilt_patches', 'debian/patches\n'), ('a/.pc/.version', '2\n'), ('a/debian/source/format', '3.0 (quilt)'), ('a/debian/patches/series', 'patch1\n'), ('a/debian/patches/patch1', TRIVIAL_PATCH)])
    tree_a.smart_add([tree_a.basedir])
    tree_a.commit('initial')
    tree_b = tree_a.controldir.sprout('b').open_workingtree()
    self.build_tree_contents([('a/debian/patches/patch1', '\n'.join(TRIVIAL_PATCH.splitlines()[:-1] + ['+d\n']))])
    quilt_push_all(tree_a)
    tree_a.smart_add([tree_a.basedir])
    tree_a.commit('apply patches')
    self.build_tree_contents([('b/debian/patches/patch1', '\n'.join(TRIVIAL_PATCH.splitlines()[:-1] + ['+c\n']))])
    quilt_push_all(tree_b)
    tree_b.commit('apply patches')
    conflicts = tree_a.merge_from_branch(tree_b.branch)
    self.assertFileEqual('--- /dev/null\t2012-01-02 01:09:10.986490031 +0100\n+++ base/a\t2012-01-02 20:03:59.710666215 +0100\n@@ -0,0 +1 @@\n<<<<<<< TREE\n+d\n=======\n+c\n>>>>>>> MERGE-SOURCE\n', 'a/debian/patches/patch1')
    self.assertPathDoesNotExist('a/a')
    self.assertEqual(1, len(conflicts))