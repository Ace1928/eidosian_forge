from dulwich.tests import TestCase
from ..graph import WorkList, _find_lcas, can_fast_forward
from ..repo import MemoryRepo
from .utils import make_commit
def test_multiple_lca(self):
    graph = {'5': ['1', '2'], '4': ['3', '1'], '3': ['2'], '2': ['0'], '1': [], '0': []}
    self.assertEqual(self.run_test(graph, ['4', '5']), {'1', '2'})