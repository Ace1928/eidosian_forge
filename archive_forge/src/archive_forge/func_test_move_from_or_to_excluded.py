import os
from io import BytesIO
import breezy
from .. import config, controldir, errors, trace
from .. import transport as _mod_transport
from ..branch import Branch
from ..bzr.bzrdir import BzrDirMetaFormat1
from ..commit import (CannotCommitSelectedFileMerge, Commit,
from ..errors import BzrError, LockContention
from ..tree import TreeChange
from . import TestCase, TestCaseWithTransport, test_foreign
from .features import SymlinkFeature
from .matchers import MatchesAncestry, MatchesTreeChanges
def test_move_from_or_to_excluded(self):
    changes = [TreeChange(('oldpath', 'newpath'), 0, (False, False), ('oldpath', 'newpath'), ('file', 'file'), (True, True))]
    self.assertEqual([], list(filter_excluded(changes, ['oldpath'])))
    self.assertEqual([], list(filter_excluded(changes, ['newpath'])))