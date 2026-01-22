from dulwich.tests import TestCase
from ..graph import WorkList, _find_lcas, can_fast_forward
from ..repo import MemoryRepo
from .utils import make_commit
def test_another_crossover(self):
    graph = {'G': ['D', 'F'], 'F': ['E', 'C'], 'D': ['C', 'E'], 'C': ['B'], 'E': ['B'], 'B': ['A'], 'A': []}
    self.assertEqual(self.run_test(graph, ['D', 'F']), {'E', 'C'})