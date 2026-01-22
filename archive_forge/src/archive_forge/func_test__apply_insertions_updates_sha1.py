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
def test__apply_insertions_updates_sha1(self):
    trans, root, contents, sha1 = self.transform_for_sha1_test()
    trans_id = trans.create_path('file1', root)
    trans.create_file(contents, trans_id, sha1=sha1)
    st_val = osutils.lstat(trans._limbo_name(trans_id))
    o_sha1, o_st_val = trans._observed_sha1s[trans_id]
    self.assertEqual(o_sha1, sha1)
    self.assertEqualStat(o_st_val, st_val)
    creation_mtime = trans._creation_mtime + 10.0
    os.utime(trans._limbo_name(trans_id), (creation_mtime, creation_mtime))
    trans.apply()
    new_st_val = osutils.lstat(self.wt.abspath('file1'))
    o_sha1, o_st_val = trans._observed_sha1s[trans_id]
    self.assertEqual(o_sha1, sha1)
    self.assertEqualStat(o_st_val, new_st_val)
    self.assertNotEqual(st_val.st_mtime, new_st_val.st_mtime)