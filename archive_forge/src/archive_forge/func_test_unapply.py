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
def test_unapply(self):
    orig_tree = self.make_branch_and_tree('source')
    self.build_tree(['source/debian/', 'source/debian/patches/'])
    self.build_tree_contents([('source/debian/patches/series', 'patch1.diff\n'), ('source/debian/patches/patch1.diff', TRIVIAL_PATCH)])
    quilt_push_all(orig_tree)
    orig_tree.smart_add([orig_tree.basedir])
    tree, target_dir = tree_unapply_patches(orig_tree)
    self.addCleanup(shutil.rmtree, target_dir)
    self.assertPathExists('source/a')
    self.assertNotEqual(tree.basedir, orig_tree.basedir)
    self.assertPathDoesNotExist(tree.abspath('a'))
    self.assertPathExists(tree.abspath('debian/patches/series'))