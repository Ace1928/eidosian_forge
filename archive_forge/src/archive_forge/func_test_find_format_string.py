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
def test_find_format_string(self):
    branch = self.make_branch('branch')
    self.assertRaises(errors.NoWorkingTree, bzrworkingtree.WorkingTreeFormatMetaDir.find_format_string, branch.controldir)
    transport = branch.controldir.get_workingtree_transport(None)
    transport.mkdir('.')
    transport.put_bytes('format', b'some format name')
    self.assertEqual(b'some format name', bzrworkingtree.WorkingTreeFormatMetaDir.find_format_string(branch.controldir))