import os
import shutil
import tempfile
import unittest
import patiencediff
from . import _patiencediff_py
def test_matching_blocks_tuples(self):
    self.assertDiffBlocks([], [], [])
    self.assertDiffBlocks([('a',), ('b',), 'c,'], [], [])
    self.assertDiffBlocks([], [('a',), ('b',), 'c,'], [])
    self.assertDiffBlocks([('a',), ('b',), 'c,'], [('a',), ('b',), 'c,'], [(0, 0, 3)])
    self.assertDiffBlocks([('a',), ('b',), 'c,'], [('a',), ('b',), 'd,'], [(0, 0, 2)])
    self.assertDiffBlocks([('d',), ('b',), 'c,'], [('a',), ('b',), 'c,'], [(1, 1, 2)])
    self.assertDiffBlocks([('d',), ('a',), ('b',), 'c,'], [('a',), ('b',), 'c,'], [(1, 0, 3)])
    self.assertDiffBlocks([('a', 'b'), ('c', 'd'), ('e', 'f')], [('a', 'b'), ('c', 'X'), ('e', 'f')], [(0, 0, 1), (2, 2, 1)])
    self.assertDiffBlocks([('a', 'b'), ('c', 'd'), ('e', 'f')], [('a', 'b'), ('c', 'dX'), ('e', 'f')], [(0, 0, 1), (2, 2, 1)])