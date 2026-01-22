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
def test_string_conflicts(self):
    tt, emerald, oz, old_dorothy, new_dorothy, munchkincity = self.get_conflicted()
    raw_conflicts = resolve_conflicts(tt)
    cooked_conflicts = list(tt.cook_conflicts(raw_conflicts))
    tt.finalize()
    conflicts_s = [str(c) for c in cooked_conflicts]
    self.assertEqual(len(cooked_conflicts), len(conflicts_s))
    if self.wt.supports_setting_file_ids():
        self.assertEqual(conflicts_s[0], 'Conflict adding file dorothy.  Moved existing file to dorothy.moved.')
        self.assertEqual(conflicts_s[1], 'Conflict adding id to dorothy.  Unversioned existing file dorothy.moved.')
        self.assertEqual(conflicts_s[2], 'Conflict adding files to munchkincity.  Created directory.')
        self.assertEqual(conflicts_s[3], 'Conflict because munchkincity is not versioned, but has versioned children.  Versioned directory.')
        self.assertEqualDiff(conflicts_s[4], "Conflict: can't delete oz because it is not empty.  Not deleting.")
        self.assertEqual(conflicts_s[5], 'Conflict because oz is not versioned, but has versioned children.  Versioned directory.')
        self.assertEqual(conflicts_s[6], 'Conflict moving oz/emeraldcity into oz/emeraldcity. Cancelled move.')
    elif self.wt.has_versioned_directories():
        self.assertEqual({'Text conflict in dorothy.moved', 'Text conflict in munchkincity', 'Text conflict in oz', 'Text conflict in oz/emeraldcity'}, {c for c in conflicts_s})
    else:
        self.assertEqual({'Text conflict in dorothy.moved', 'Text conflict in oz', 'Text conflict in oz/emeraldcity'}, {c for c in conflicts_s})