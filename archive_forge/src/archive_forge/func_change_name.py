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
def change_name(self, trans_ids, base=None, this=None, other=None):
    for val, tt, trans_id in ((base, self.base_tt, trans_ids[0]), (this, self.this_tt, trans_ids[1]), (other, self.other_tt, trans_ids[2])):
        if val is None:
            continue
        parent_id = tt.final_parent(trans_id)
        tt.adjust_path(val, parent_id, trans_id)