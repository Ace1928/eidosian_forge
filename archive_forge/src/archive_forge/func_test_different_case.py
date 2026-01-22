from ...revision import Revision
from ...tests import TestCase, TestCaseWithTransport
from .cmds import collapse_by_person, get_revisions_and_committers
def test_different_case(self):
    wt = self.make_branch_and_tree('.')
    wt.commit(message='1', committer='Fero', rev_id=b'1')
    wt.commit(message='2', committer='Fero', rev_id=b'2')
    wt.commit(message='3', committer='FERO', rev_id=b'3')
    revs, committers = get_revisions_and_committers(wt.branch.repository, [b'1', b'2', b'3'])
    self.assertEqual({('Fero', ''): ('Fero', ''), ('FERO', ''): ('Fero', '')}, committers)
    self.assertEqual([b'1', b'2', b'3'], sorted([r.revision_id for r in revs]))