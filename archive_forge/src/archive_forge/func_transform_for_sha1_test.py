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
def transform_for_sha1_test(self):
    trans, root = self.transform()
    if getattr(self.wt, '_observed_sha1', None) is None:
        raise tests.TestNotApplicable('wt format does not use _observed_sha1')
    self.wt.lock_tree_write()
    self.addCleanup(self.wt.unlock)
    contents = [b'just some content\n']
    sha1 = osutils.sha_strings(contents)
    trans._creation_mtime = time.time() - 20.0
    return (trans, root, contents, sha1)