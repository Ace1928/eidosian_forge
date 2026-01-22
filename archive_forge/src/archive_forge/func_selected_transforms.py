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
def selected_transforms(self, this, base, other):
    pairs = [(this, self.this_tt), (base, self.base_tt), (other, self.other_tt)]
    return [(v, tt) for v, tt in pairs if v is not None]