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
def test_convenience(self):
    transform, root = self.transform()
    self.wt.lock_tree_write()
    self.addCleanup(self.wt.unlock)
    transform.new_file('name', root, [b'contents'], b'my_pretties', True)
    oz = transform.new_directory('oz', root, b'oz-id')
    dorothy = transform.new_directory('dorothy', oz, b'dorothy-id')
    transform.new_file('toto', dorothy, [b'toto-contents'], b'toto-id', False)
    self.assertEqual(len(transform.find_raw_conflicts()), 0)
    transform.apply()
    self.assertRaises(ReusingTransform, transform.find_raw_conflicts)
    with open(self.wt.abspath('name')) as f:
        self.assertEqual('contents', f.read())
    self.assertIs(self.wt.is_executable('name'), True)
    self.assertTrue(self.wt.is_versioned('name'))
    self.assertTrue(self.wt.is_versioned('oz'))
    self.assertTrue(self.wt.is_versioned('oz/dorothy'))
    self.assertTrue(self.wt.is_versioned('oz/dorothy/toto'))
    if self.wt.supports_setting_file_ids():
        self.assertEqual(self.wt.path2id('name'), b'my_pretties')
        self.assertEqual(self.wt.path2id('oz'), b'oz-id')
        self.assertEqual(self.wt.path2id('oz/dorothy'), b'dorothy-id')
        self.assertEqual(self.wt.path2id('oz/dorothy/toto'), b'toto-id')
    with self.wt.get_file('oz/dorothy/toto') as f:
        self.assertEqual(b'toto-contents', f.read())
    self.assertIs(self.wt.is_executable('oz/dorothy/toto'), False)