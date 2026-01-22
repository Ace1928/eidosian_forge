import codecs
import errno
import os
import sys
import time
from io import BytesIO, StringIO
import fastbencode as bencode
from .. import filters, osutils
from .. import revision as _mod_revision
from .. import rules, tests, trace, transform, urlutils
from ..bzr import generate_ids
from ..bzr.conflicts import (DeletingParent, DuplicateEntry, DuplicateID,
from ..controldir import ControlDir
from ..diff import show_diff_trees
from ..errors import (DuplicateKey, ExistingLimbo, ExistingPendingDeletion,
from ..merge import Merge3Merger, Merger
from ..mutabletree import MutableTree
from ..osutils import file_kind, pathjoin
from ..transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ..transport import FileExists
from . import TestCaseInTempDir, TestSkipped, features
from .features import HardlinkFeature, SymlinkFeature
def test_post_commit_hooks(self):
    calls = []

    def record_post_transform(tree, tt):
        calls.append((tree, tt))
    MutableTree.hooks.install_named_hook('post_transform', record_post_transform, 'Post transform')
    transform, root = self.transform()
    old_root_id = transform.tree_file_id(root)
    transform.apply()
    self.assertEqual(old_root_id, self.wt.path2id(''))
    self.assertEqual([(self.wt, transform)], calls)