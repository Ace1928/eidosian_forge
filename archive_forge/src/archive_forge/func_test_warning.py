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
def test_warning(self):
    self.enable_hooks()
    warnings = []

    def warning(*args):
        if len(args) > 1:
            warnings.append(args[0] % args[1:])
        else:
            warnings.append(args[0])
    _warning = trace.warning
    trace.warning = warning
    self.addCleanup(setattr, trace, 'warning', _warning)
    tree = self.make_branch_and_tree('source')
    self.build_tree(['source/debian/', 'source/debian/patches/', 'source/debian/source/'])
    self.build_tree_contents([('source/debian/patches/series', 'patch1\n'), ('source/debian/patches/patch1', TRIVIAL_PATCH)])
    quilt_push_all(tree)
    tree.smart_add([tree.basedir])
    tree.commit('initial')
    self.assertEqual([], warnings)
    self.assertPathExists('source/.pc/applied-patches')
    self.assertPathExists('source/a')
    self.build_tree_contents([('source/debian/source/format', '3.0 (quilt)'), ('source/debian/patches/series', 'patch1\npatch2\n'), ('source/debian/patches/patch2', '--- /dev/null\t2012-01-02 01:09:10.986490031 +0100\n+++ base/b\t2012-01-02 20:03:59.710666215 +0100\n@@ -0,0 +1 @@\n+a\n')])
    tree.smart_add([tree.basedir])
    tree.commit('foo')
    self.assertEqual(['Committing with 1 patches applied and 1 patches unapplied.'], warnings)
    self.assertPathExists('source/.pc/applied-patches')
    self.assertPathExists('source/a')
    self.assertPathDoesNotExist('source/b')