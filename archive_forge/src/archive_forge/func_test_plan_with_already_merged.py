from ....conflicts import ConflictList
from ....errors import ConflictsInTree, UnknownFormatError
from ....graph import DictParentsProvider, Graph
from ....revision import NULL_REVISION
from ....tests import TestCase, TestCaseWithTransport
from ....tests.matchers import RevisionHistoryMatches
from ....transport import NoSuchFile
from ..rebase import (REBASE_CURRENT_REVID_FILENAME, REBASE_PLAN_FILENAME,
def test_plan_with_already_merged(self):
    """We need to use a merge base that makes sense.

        A
        | \\
        B  D
        | \\|
        C  E

        Rebasing E on C should result in:

        A -> B -> C -> D -> E

        with a plan of:

        D -> (D', [C])
        E -> (E', [D', C])
        """
    parents_map = {'A': (), 'B': ('A',), 'C': ('B',), 'D': ('A',), 'E': ('D', 'B')}
    graph = Graph(DictParentsProvider(parents_map))
    self.assertEqual({'D': ("D'", ('C',)), 'E': ("E'", ("D'",))}, generate_simple_plan(['D', 'E'], 'D', None, 'C', graph, lambda y, _: y + "'"))