import os
from io import BytesIO
from .. import (conflicts, errors, symbol_versioning, trace, transport,
from ..bzr import bzrdir
from ..bzr import conflicts as _mod_bzr_conflicts
from ..bzr import workingtree as bzrworkingtree
from ..bzr import workingtree_3, workingtree_4
from ..lock import write_locked
from ..lockdir import LockDir
from ..tree import TreeDirectory, TreeEntry, TreeFile, TreeLink
from . import TestCase, TestCaseWithTransport, TestSkipped
from .features import SymlinkFeature
def test_find_format_no_tree(self):
    dir = bzrdir.BzrDirMetaFormat1().initialize('.')
    self.assertRaises(errors.NoWorkingTree, bzrworkingtree.WorkingTreeFormatMetaDir.find_format, dir)