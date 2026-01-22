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
def change_perms(self, trans_ids, base=None, this=None, other=None):
    for trans_id, (executability, tt) in zip(trans_ids, self.selected_transforms(this, base, other)):
        tt.set_executability(None, trans_id)
        tt.set_executability(executability, trans_id)