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
def test_applied(self):
    self.enable_hooks()
    tree = self.make_branch_and_tree('source')
    tree.get_config_stack().set('quilt.commit_policy', 'applied')
    self.build_tree(['source/debian/', 'source/debian/patches/', 'source/debian/source/'])
    self.build_tree_contents([('source/debian/source/format', '3.0 (quilt)'), ('source/debian/patches/series', 'patch1\n'), ('source/debian/patches/patch1', TRIVIAL_PATCH)])
    self.assertPathDoesNotExist('source/.pc/applied-patches')
    self.assertPathDoesNotExist('source/a')
    tree.smart_add([tree.basedir])
    tree.commit('foo')
    self.assertPathExists('source/.pc/applied-patches')
    self.assertPathExists('source/a')