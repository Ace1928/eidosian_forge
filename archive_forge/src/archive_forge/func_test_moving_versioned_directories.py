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
def test_moving_versioned_directories(self):
    create, root = self.transform()
    kansas = create.new_directory('kansas', root, b'kansas-id')
    create.new_directory('house', kansas, b'house-id')
    create.new_directory('oz', root, b'oz-id')
    create.apply()
    cyclone, root = self.transform()
    oz = cyclone.trans_id_tree_path('oz')
    house = cyclone.trans_id_tree_path('house')
    cyclone.adjust_path('house', oz, house)
    cyclone.apply()