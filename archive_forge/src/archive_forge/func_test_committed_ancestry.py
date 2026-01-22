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
def test_committed_ancestry(self):
    """Test commit appends revisions to ancestry."""
    wt = self.make_branch_and_tree('.')
    b = wt.branch
    rev_ids = []
    for i in range(4):
        with open('hello', 'w') as f:
            f.write(str(i) * 4 + '\n')
        if i == 0:
            wt.add(['hello'], ids=[b'hello-id'])
        rev_id = b'test@rev-%d' % (i + 1)
        rev_ids.append(rev_id)
        wt.commit(message='rev %d' % (i + 1), rev_id=rev_id)
    for i in range(4):
        self.assertThat(rev_ids[:i + 1], MatchesAncestry(b.repository, rev_ids[i]))