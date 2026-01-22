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
def test_iter_changes_modifications(self):
    transform, root = self.transform()
    transform.new_file('old', root, [b'blah'], b'id-1')
    transform.new_file('new', root, [b'blah'])
    transform.new_directory('subdir', root, b'subdir-id')
    transform.apply()
    transform, root = self.transform()
    try:
        old = transform.trans_id_tree_path('old')
        subdir = transform.trans_id_tree_path('subdir')
        new = transform.trans_id_tree_path('new')
        self.assertTreeChanges(transform, [])
        transform.delete_contents(old)
        self.assertTreeChanges(transform, [TreeChange(('old', 'old'), True, (True, True), ('old', 'old'), ('file', None), (False, False), False)])
        transform.create_file([b'blah'], old)
        self.assertTreeChanges(transform, [TreeChange(('old', 'old'), True, (True, True), ('old', 'old'), ('file', 'file'), (False, False), False)])
        transform.cancel_deletion(old)
        self.assertTreeChanges(transform, [TreeChange(('old', 'old'), True, (True, True), ('old', 'old'), ('file', 'file'), (False, False), False)])
        transform.cancel_creation(old)
        self.assertTreeChanges(transform, [])
        transform.unversion_file(old)
        transform.version_file(new, file_id=b'id-1')
        transform.adjust_path('old', root, new)
        if transform._tree.supports_setting_file_ids():
            self.assertTreeChanges(transform, [TreeChange(('old', 'old'), True, (True, True), ('old', 'old'), ('file', 'file'), (False, False), False)])
        else:
            self.assertTreeChanges(transform, [TreeChange((None, 'old'), False, (False, True), (None, 'old'), (None, 'file'), (False, False), False), TreeChange(('old', None), False, (True, False), ('old', 'old'), ('file', 'file'), (False, False), False)])
        transform.cancel_versioning(new)
        transform._removed_id = set()
        self.assertTreeChanges(transform, [])
        transform.set_executability(True, old)
        self.assertTreeChanges(transform, [TreeChange(('old', 'old'), False, (True, True), ('old', 'old'), ('file', 'file'), (False, True), False)])
        transform.set_executability(None, old)
        self.assertTreeChanges(transform, [])
        transform.adjust_path('new', root, old)
        transform._new_parent = {}
        self.assertTreeChanges(transform, [TreeChange(('old', 'new'), False, (True, True), ('old', 'new'), ('file', 'file'), (False, False), False)])
        transform._new_name = {}
        self.assertTreeChanges(transform, [])
        transform.adjust_path('new', subdir, old)
        transform._new_name = {}
        self.assertTreeChanges(transform, [TreeChange(('old', 'subdir/old'), False, (True, True), ('old', 'old'), ('file', 'file'), (False, False), False)])
        transform._new_path = {}
    finally:
        transform.finalize()