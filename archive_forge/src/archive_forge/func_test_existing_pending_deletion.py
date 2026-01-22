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
def test_existing_pending_deletion(self):
    transform, root = self.transform()
    deletion_path = self._limbodir = urlutils.local_path_from_url(transform._tree._transport.abspath('pending-deletion'))
    os.mkdir(pathjoin(deletion_path, 'blocking-directory'))
    self.assertRaises(ImmortalPendingDeletion, transform.apply)
    self.assertRaises(LockError, self.wt.unlock)
    self.assertRaises(ExistingPendingDeletion, self.transform)