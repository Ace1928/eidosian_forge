from pyomo.common import unittest
from pyomo.contrib.solver.solution import SolutionLoaderBase, PersistentSolutionLoader
@unittest.mock.patch.multiple(SolutionLoaderBase, __abstractmethods__=set())
def test_solution_loader_base(self):
    self.instance = SolutionLoaderBase()
    self.assertEqual(self.instance.get_primals(), None)
    with self.assertRaises(NotImplementedError):
        self.instance.get_duals()
    with self.assertRaises(NotImplementedError):
        self.instance.get_reduced_costs()