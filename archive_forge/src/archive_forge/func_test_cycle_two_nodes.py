import gyp.input
import unittest
def test_cycle_two_nodes(self):
    self._create_dependency(self.nodes['a'], self.nodes['b'])
    self._create_dependency(self.nodes['b'], self.nodes['a'])
    self.assertEqual([[self.nodes['a'], self.nodes['b'], self.nodes['a']]], self.nodes['a'].FindCycles())
    self.assertEqual([[self.nodes['b'], self.nodes['a'], self.nodes['b']]], self.nodes['b'].FindCycles())