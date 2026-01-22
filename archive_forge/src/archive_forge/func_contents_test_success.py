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
def contents_test_success(self, merge_factory):
    builder = MergeBuilder(getcwd())
    name1 = builder.add_file(builder.root(), 'name1', b'text1', True, file_id=b'1')
    builder.change_contents(name1, other=b'text4')
    name3 = builder.add_file(builder.root(), 'name3', b'text2', False, file_id=b'2')
    builder.change_contents(name3, base=b'text5')
    builder.add_file(builder.root(), 'name5', b'text3', True, file_id=b'3')
    name6 = builder.add_file(builder.root(), 'name6', b'text4', True, file_id=b'4')
    builder.remove_file(name6, base=True)
    name7 = builder.add_file(builder.root(), 'name7', b'a\nb\nc\nd\ne\nf\n', True, file_id=b'5')
    builder.change_contents(name7, other=b'a\nz\nc\nd\ne\nf\n', this=b'a\nb\nc\nd\ne\nz\n')
    conflicts = builder.merge(merge_factory)
    try:
        self.assertEqual([], conflicts)
        self.assertEqual(b'text4', builder.this.get_file('name1').read())
        self.assertEqual(b'text2', builder.this.get_file('name3').read())
        self.assertEqual(b'a\nz\nc\nd\ne\nz\n', builder.this.get_file('name7').read())
        self.assertTrue(builder.this.is_executable('name1'))
        self.assertFalse(builder.this.is_executable('name3'))
        self.assertTrue(builder.this.is_executable('name5'))
    except:
        builder.unlock()
        raise
    return builder