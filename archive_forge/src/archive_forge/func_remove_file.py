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
def remove_file(self, trans_ids, base=False, this=False, other=False):
    for trans_id, (option, tt) in zip(trans_ids, self.selected_transforms(this, base, other)):
        if option is True:
            tt.cancel_creation(trans_id)
            tt.cancel_versioning(trans_id)
            tt.set_executability(None, trans_id)