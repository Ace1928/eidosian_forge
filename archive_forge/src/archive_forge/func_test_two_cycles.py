import gyp.input
import unittest
def test_two_cycles(self):
    self._create_dependency(self.nodes['a'], self.nodes['b'])
    self._create_dependency(self.nodes['b'], self.nodes['a'])
    self._create_dependency(self.nodes['b'], self.nodes['c'])
    self._create_dependency(self.nodes['c'], self.nodes['b'])
    cycles = self.nodes['a'].FindCycles()
    self.assertTrue([self.nodes['a'], self.nodes['b'], self.nodes['a']] in cycles)
    self.assertTrue([self.nodes['b'], self.nodes['c'], self.nodes['b']] in cycles)
    self.assertEqual(2, len(cycles))