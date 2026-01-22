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
def test_selective_delete(self):
    """Selective commit in tree with deletions"""
    wt = self.make_branch_and_tree('.')
    b = wt.branch
    with open('hello', 'w') as f:
        f.write('hello')
    with open('buongia', 'w') as f:
        f.write('buongia')
    wt.add(['hello', 'buongia'], ids=[b'hello-id', b'buongia-id'])
    wt.commit(message='add files', rev_id=b'test@rev-1')
    os.remove('hello')
    with open('buongia', 'w') as f:
        f.write('new text')
    wt.commit(message='update text', specific_files=['buongia'], allow_pointless=False, rev_id=b'test@rev-2')
    wt.commit(message='remove hello', specific_files=['hello'], allow_pointless=False, rev_id=b'test@rev-3')
    eq = self.assertEqual
    eq(b.revno(), 3)
    tree2 = b.repository.revision_tree(b'test@rev-2')
    tree2.lock_read()
    self.addCleanup(tree2.unlock)
    self.assertTrue(tree2.has_filename('hello'))
    self.assertEqual(tree2.get_file_text('hello'), b'hello')
    self.assertEqual(tree2.get_file_text('buongia'), b'new text')
    tree3 = b.repository.revision_tree(b'test@rev-3')
    tree3.lock_read()
    self.addCleanup(tree3.unlock)
    self.assertFalse(tree3.has_filename('hello'))
    self.assertEqual(tree3.get_file_text('buongia'), b'new text')