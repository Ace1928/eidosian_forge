from ....conflicts import ConflictList
from ....errors import ConflictsInTree, UnknownFormatError
from ....graph import DictParentsProvider, Graph
from ....revision import NULL_REVISION
from ....tests import TestCase, TestCaseWithTransport
from ....tests.matchers import RevisionHistoryMatches
from ....transport import NoSuchFile
from ..rebase import (REBASE_CURRENT_REVID_FILENAME, REBASE_PLAN_FILENAME,
def test_already_merged(self):
    """We need to use a merge base that makes sense.

        A
        | \\
        B  D
        | \\|
        C  E

        Rebasing E on C should result in:

        A -> B -> C -> D' -> E'

        Ancestry:
        A:
        B: A
        C: A, B
        D: A
        E: A, B, D
        D': A, B, C
        E': A, B, C, D'

        """
    oldwt = self.make_branch_and_tree('old')
    self.build_tree(['old/afile'])
    with open('old/afile', 'w') as f:
        f.write('A\n' * 10)
    oldwt.add(['afile'])
    oldwt.commit('base', rev_id=b'A')
    newwt = oldwt.controldir.sprout('new').open_workingtree()
    with open('old/afile', 'w') as f:
        f.write('A\n' * 10 + 'B\n')
    oldwt.commit('bla', rev_id=b'B')
    with open('old/afile', 'w') as f:
        f.write('A\n' * 10 + 'C\n')
    oldwt.commit('bla', rev_id=b'C')
    self.build_tree(['new/bfile'])
    newwt.add(['bfile'])
    with open('new/bfile', 'w') as f:
        f.write('D\n')
    newwt.commit('bla', rev_id=b'D')
    with open('new/afile', 'w') as f:
        f.write('E\n' + 'A\n' * 10 + 'B\n')
    with open('new/bfile', 'w') as f:
        f.write('D\nE\n')
    newwt.add_pending_merge(b'B')
    newwt.commit('bla', rev_id=b'E')
    newwt.branch.repository.fetch(oldwt.branch.repository)
    newwt.lock_write()
    replayer = WorkingTreeRevisionRewriter(newwt, RebaseState1(newwt))
    replayer(b'D', b"D'", [b'C'])
    newwt.unlock()
    oldrev = newwt.branch.repository.get_revision(b'D')
    newrev = newwt.branch.repository.get_revision(b"D'")
    self.assertEqual([b'C'], newrev.parent_ids)
    newwt.lock_write()
    replayer = WorkingTreeRevisionRewriter(newwt, RebaseState1(newwt))
    self.assertRaises(ConflictsInTree, replayer, b'E', b"E'", [b"D'"])
    newwt.unlock()
    with open('new/afile') as f:
        self.assertEqual('E\n' + 'A\n' * 10 + 'C\n', f.read())
    newwt.set_conflicts([])
    oldrev = newwt.branch.repository.get_revision(b'E')
    replayer.commit_rebase(oldrev, b"E'")
    newrev = newwt.branch.repository.get_revision(b"E'")
    self.assertEqual([b"D'"], newrev.parent_ids)
    self.assertThat(newwt.branch, RevisionHistoryMatches([b'A', b'B', b'C', b"D'", b"E'"]))