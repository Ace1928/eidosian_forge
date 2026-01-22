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
def test_change_name(self):
    """Test renames"""
    builder = MergeBuilder(getcwd())
    name1 = builder.add_file(builder.root(), 'name1', b'hello1', True, file_id=b'1')
    builder.change_name(name1, other='name2')
    name3 = builder.add_file(builder.root(), 'name3', b'hello2', True, file_id=b'2')
    builder.change_name(name3, base='name4')
    name5 = builder.add_file(builder.root(), 'name5', b'hello3', True, file_id=b'3')
    builder.change_name(name5, this='name6')
    builder.merge()
    builder.cleanup()
    builder = MergeBuilder(getcwd())
    name1 = builder.add_file(builder.root(), 'name1', b'hello1', False, file_id=b'1')
    builder.change_name(name1, other='name2', this='name3')
    conflicts = builder.merge()
    self.assertEqual(conflicts, [PathConflict('name3', 'name2', b'1')])
    builder.cleanup()