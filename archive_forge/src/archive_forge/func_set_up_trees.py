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
def set_up_trees(self):
    this = self.make_branch_and_tree('this')
    this.commit('rev1', rev_id=b'rev1')
    other = this.controldir.sprout('other').open_workingtree()
    this.commit('rev2a', rev_id=b'rev2a')
    other.commit('rev2b', rev_id=b'rev2b')
    return (this, other)