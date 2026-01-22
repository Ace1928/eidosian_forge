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
def test_symlink_conflicts(self):
    if sys.platform != 'win32':
        builder = MergeBuilder(getcwd())
        name2 = builder.add_symlink(builder.root(), 'name2', 'target1', file_id=b'2')
        builder.change_target(name2, other='target4', base='text3')
        conflicts = builder.merge()
        self.assertEqual(conflicts, [ContentsConflict('name2', file_id=b'2')])
        builder.cleanup()