import gyp.input
import unittest
def test_no_cycle_line(self):
    self._create_dependency(self.nodes['a'], self.nodes['b'])
    self._create_dependency(self.nodes['b'], self.nodes['c'])
    self._create_dependency(self.nodes['c'], self.nodes['d'])
    for label, node in self.nodes.items():
        self.assertEqual([], node.FindCycles())