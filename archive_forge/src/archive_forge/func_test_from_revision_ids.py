import os
import sys
import tempfile
import breezy
from .. import controldir, errors, merge_directive, osutils
from ..bzr import generate_ids
from ..bzr.conflicts import ContentsConflict, PathConflict, TextConflict
from ..merge import Diff3Merger, Merge3Merger, Merger, WeaveMerger
from ..osutils import getcwd, pathjoin
from ..workingtree import WorkingTree
from . import TestCaseWithTransport, TestSkipped, features
def test_from_revision_ids(self):
    this, other = self.set_up_trees()
    self.assertRaises(errors.NoSuchRevision, Merger.from_revision_ids, this, b'rev2b')
    this.lock_write()
    self.addCleanup(this.unlock)
    merger = Merger.from_revision_ids(this, b'rev2b', other_branch=other.branch)
    self.assertEqual(b'rev2b', merger.other_rev_id)
    self.assertEqual(b'rev1', merger.base_rev_id)
    merger = Merger.from_revision_ids(this, b'rev2b', b'rev2a', other_branch=other.branch)
    self.assertEqual(b'rev2a', merger.base_rev_id)