from ....conflicts import ConflictList
from ....errors import ConflictsInTree, UnknownFormatError
from ....graph import DictParentsProvider, Graph
from ....revision import NULL_REVISION
from ....tests import TestCase, TestCaseWithTransport
from ....tests.matchers import RevisionHistoryMatches
from ....transport import NoSuchFile
from ..rebase import (REBASE_CURRENT_REVID_FILENAME, REBASE_PLAN_FILENAME,
def test_simple_plan_creator(self):
    wt = self.make_branch_and_tree('.')
    b = wt.branch
    with open('hello', 'w') as f:
        f.write('hello world')
    wt.add('hello')
    wt.commit(message='add hello', rev_id=b'bla')
    with open('hello', 'w') as f:
        f.write('world')
    wt.commit(message='change hello', rev_id=b'bloe')
    wt.set_last_revision(b'bla')
    b.generate_revision_history(b'bla')
    with open('hello', 'w') as f:
        f.write('world')
    wt.commit(message='change hello', rev_id=b'bla2')
    b.repository.lock_read()
    graph = b.repository.get_graph()
    self.assertEqual({b'bla2': (b'newbla2', (b'bloe',))}, generate_simple_plan(graph.find_difference(b.last_revision(), b'bla')[0], b'bla2', None, b'bloe', graph, lambda y, _: b'new' + y))
    b.repository.unlock()