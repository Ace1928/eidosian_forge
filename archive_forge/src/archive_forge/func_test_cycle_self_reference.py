import gyp.input
import unittest
def test_cycle_self_reference(self):
    self._create_dependency(self.nodes['a'], self.nodes['a'])
    self.assertEqual([[self.nodes['a'], self.nodes['a']]], self.nodes['a'].FindCycles())