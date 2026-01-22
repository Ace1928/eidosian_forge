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
def prepare_for_merging(self):
    this, other = self.set_up_trees()
    other.commit('rev3', rev_id=b'rev3')
    this.lock_write()
    self.addCleanup(this.unlock)
    return (this, other)