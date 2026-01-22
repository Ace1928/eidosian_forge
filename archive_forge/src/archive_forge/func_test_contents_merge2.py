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
def test_contents_merge2(self):
    """Test diff3 merging"""
    if sys.platform == 'win32':
        raise TestSkipped('diff3 does not have --binary flag and therefore always fails on win32')
    try:
        self.do_contents_test(Diff3Merger)
    except errors.NoDiff3:
        raise TestSkipped('diff3 not available')