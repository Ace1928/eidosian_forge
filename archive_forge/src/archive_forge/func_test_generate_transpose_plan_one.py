from ....conflicts import ConflictList
from ....errors import ConflictsInTree, UnknownFormatError
from ....graph import DictParentsProvider, Graph
from ....revision import NULL_REVISION
from ....tests import TestCase, TestCaseWithTransport
from ....tests.matchers import RevisionHistoryMatches
from ....transport import NoSuchFile
from ..rebase import (REBASE_CURRENT_REVID_FILENAME, REBASE_PLAN_FILENAME,
def test_generate_transpose_plan_one(self):
    graph = Graph(DictParentsProvider({'bla': ('bloe',), 'bloe': (), 'lala': ()}))
    self.assertEqual({'bla': ('newbla', ('lala',))}, generate_transpose_plan(graph.iter_ancestry(['bla', 'bloe']), {'bloe': 'lala'}, graph, lambda y, _: 'new' + y))