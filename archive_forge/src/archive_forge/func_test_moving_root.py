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
def test_moving_root(self):
    create, root = self.transform()
    fun = create.new_directory('fun', root, b'fun-id')
    create.new_directory('sun', root, b'sun-id')
    create.new_directory('moon', root, b'moon')
    create.apply()
    transform, root = self.transform()
    transform.adjust_root_path('oldroot', fun)
    new_root = transform.trans_id_tree_path('')
    transform.version_file(new_root, file_id=b'new-root')
    transform.apply()