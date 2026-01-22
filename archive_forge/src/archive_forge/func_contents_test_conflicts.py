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
def contents_test_conflicts(self, merge_factory):
    builder = MergeBuilder(getcwd())
    name1 = builder.add_file(builder.root(), 'name1', b'text1', True, file_id=b'1')
    builder.change_contents(name1, other=b'text4', this=b'text3')
    name2 = builder.add_file(builder.root(), 'name2', b'text1', True, file_id=b'2')
    builder.change_contents(name2, other=b'\x00', this=b'text3')
    name3 = builder.add_file(builder.root(), 'name3', b'text5', False, file_id=b'3')
    builder.change_perms(name3, this=True)
    builder.change_contents(name3, this=b'moretext')
    builder.remove_file(name3, other=True)
    conflicts = builder.merge(merge_factory)
    self.assertEqual(conflicts, [TextConflict('name1', file_id=b'1'), ContentsConflict('name2', file_id=b'2'), ContentsConflict('name3', file_id=b'3')])
    with builder.this.get_file(builder.this.id2path(b'2')) as f:
        self.assertEqual(f.read(), b'\x00')
    builder.cleanup()