from pyomo.common import unittest
from pyomo.contrib.solver.solution import SolutionLoaderBase, PersistentSolutionLoader
def test_abstract_member_list(self):
    member_list = list(PersistentSolutionLoader('ipopt').__abstractmethods__)
    self.assertEqual(member_list, [])