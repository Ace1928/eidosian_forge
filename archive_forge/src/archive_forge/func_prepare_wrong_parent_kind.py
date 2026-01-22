import errno
import os
import sys
import time
from io import BytesIO
from breezy.bzr.transform import resolve_checkout
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from ... import osutils, tests, trace, transform, urlutils
from ...bzr.conflicts import (DeletingParent, DuplicateEntry, DuplicateID,
from ...errors import (DuplicateKey, ExistingLimbo, ExistingPendingDeletion,
from ...osutils import file_kind, pathjoin
from ...transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ...transport import FileExists
from ...tree import TreeChange
from .. import TestSkipped, features
from ..features import HardlinkFeature, SymlinkFeature
def prepare_wrong_parent_kind(self):
    tt, root = self.transform()
    tt.new_file('parent', root, [b'contents'], b'parent-id')
    tt.apply()
    tt, root = self.transform()
    parent_id = tt.trans_id_tree_path('parent')
    tt.new_file('child,', parent_id, [b'contents2'], b'file-id')
    return tt