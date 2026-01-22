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
def test_link_fails_if_execute_bit_changed(self):
    """If the file to be linked has modified execute bit, don't link."""
    tt = self.child_tree.transform()
    try:
        trans_id = tt.trans_id_tree_path('foo')
        tt.set_executability(True, trans_id)
        tt.apply()
    finally:
        tt.finalize()
    transform.link_tree(self.child_tree, self.parent_tree)
    self.assertFalse(self.hardlinked())