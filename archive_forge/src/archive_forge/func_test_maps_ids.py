from ....conflicts import ConflictList
from ....errors import ConflictsInTree, UnknownFormatError
from ....graph import DictParentsProvider, Graph
from ....revision import NULL_REVISION
from ....tests import TestCase, TestCaseWithTransport
from ....tests.matchers import RevisionHistoryMatches
from ....transport import NoSuchFile
from ..rebase import (REBASE_CURRENT_REVID_FILENAME, REBASE_PLAN_FILENAME,
def test_maps_ids(self):
    wt = self.make_branch_and_tree('old')
    wt.commit('base', rev_id=b'base')
    self.build_tree(['old/afile'])
    wt.add(['afile'], ids=[b'originalid'])
    wt.commit('bla', rev_id=b'oldparent')
    with open('old/afile', 'w') as f:
        f.write('bloe')
    wt.commit('bla', rev_id=b'oldcommit')
    oldrepos = wt.branch.repository
    wt = self.make_branch_and_tree('new')
    self.build_tree(['new/afile'])
    wt.add(['afile'], ids=[b'newid'])
    wt.commit('bla', rev_id=b'newparent')
    wt.branch.repository.fetch(oldrepos)
    wt.branch.repository.lock_write()
    CommitBuilderRevisionRewriter(wt.branch.repository)(b'oldcommit', b'newcommit', (b'newparent',))
    wt.branch.repository.unlock()
    oldrev = wt.branch.repository.get_revision(b'oldcommit')
    newrev = wt.branch.repository.get_revision(b'newcommit')
    self.assertEqual([b'newparent'], newrev.parent_ids)
    self.assertEqual(b'newcommit', newrev.revision_id)
    self.assertEqual(oldrev.committer, newrev.committer)
    self.assertEqual(oldrev.timestamp, newrev.timestamp)
    self.assertEqual(oldrev.timezone, newrev.timezone)
    tree = wt.branch.repository.revision_tree(b'newcommit')
    self.assertEqual(b'newid', tree.path2id('afile'))
    self.assertEqual(b'newcommit', tree.get_file_revision('afile'))