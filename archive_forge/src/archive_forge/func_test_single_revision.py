from ....conflicts import ConflictList
from ....errors import ConflictsInTree, UnknownFormatError
from ....graph import DictParentsProvider, Graph
from ....revision import NULL_REVISION
from ....tests import TestCase, TestCaseWithTransport
from ....tests.matchers import RevisionHistoryMatches
from ....transport import NoSuchFile
from ..rebase import (REBASE_CURRENT_REVID_FILENAME, REBASE_PLAN_FILENAME,
def test_single_revision(self):
    wt = self.make_branch_and_tree('.')
    self.build_tree(['afile'])
    wt.add(['afile'])
    wt.commit('bla', rev_id=b'oldcommit')
    wt.branch.repository.lock_write()
    CommitBuilderRevisionRewriter(wt.branch.repository)(b'oldcommit', b'newcommit', ())
    wt.branch.repository.unlock()
    oldrev = wt.branch.repository.get_revision(b'oldcommit')
    newrev = wt.branch.repository.get_revision(b'newcommit')
    self.assertEqual([], newrev.parent_ids)
    self.assertEqual(b'newcommit', newrev.revision_id)
    self.assertEqual(oldrev.committer, newrev.committer)
    self.assertEqual(oldrev.timestamp, newrev.timestamp)
    self.assertEqual(oldrev.timezone, newrev.timezone)
    tree = wt.branch.repository.revision_tree(b'newcommit')
    self.assertEqual(b'newcommit', tree.get_file_revision('afile'))