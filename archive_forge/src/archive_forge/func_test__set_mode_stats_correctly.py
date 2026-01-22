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
def test__set_mode_stats_correctly(self):
    """_set_mode stats to determine file mode."""
    if sys.platform == 'win32':
        raise TestSkipped('chmod has no effect on win32')
    stat_paths = []
    real_stat = os.stat

    def instrumented_stat(path):
        stat_paths.append(path)
        return real_stat(path)
    transform, root = self.transform()
    bar1_id = transform.new_file('bar', root, [b'bar contents 1\n'], file_id=b'bar-id-1', executable=False)
    transform.apply()
    transform, root = self.transform()
    bar1_id = transform.trans_id_tree_path('bar')
    bar2_id = transform.trans_id_tree_path('bar2')
    try:
        os.stat = instrumented_stat
        transform.create_file([b'bar2 contents\n'], bar2_id, mode_id=bar1_id)
    finally:
        os.stat = real_stat
        transform.finalize()
    bar1_abspath = self.wt.abspath('bar')
    self.assertEqual([bar1_abspath], stat_paths)