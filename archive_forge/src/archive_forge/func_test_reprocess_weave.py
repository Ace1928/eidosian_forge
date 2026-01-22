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
def test_reprocess_weave(self):
    builder = MergeBuilder(getcwd())
    blah = builder.add_file(builder.root(), 'blah', b'a', False, file_id=b'a')
    builder.change_contents(blah, this=b'b\nc\nd\ne\n', other=b'z\nc\nd\ny\n')
    builder.merge(WeaveMerger, reprocess=True)
    expected = b'<<<<<<< TREE\nb\n=======\nz\n>>>>>>> MERGE-SOURCE\nc\nd\n<<<<<<< TREE\ne\n=======\ny\n>>>>>>> MERGE-SOURCE\n'
    self.assertEqualDiff(builder.this.get_file_text('blah'), expected)
    builder.cleanup()