import unittest
from collections import defaultdict
from nltk.translate import AlignedSent, IBMModel
from nltk.translate.ibm_model import AlignmentInfo
def test_neighboring_sets_neighbor_alignment_info(self):
    a_info = AlignmentInfo((0, 3, 2), (None, 'des', 'œufs', 'verts'), ('UNUSED', 'green', 'eggs'), [[], [], [2], [1]])
    ibm_model = IBMModel([])
    neighbors = ibm_model.neighboring(a_info)
    for neighbor in neighbors:
        if neighbor.alignment == (0, 2, 2):
            moved_alignment = neighbor
        elif neighbor.alignment == (0, 3, 2):
            swapped_alignment = neighbor
    self.assertEqual(moved_alignment.cepts, [[], [], [1, 2], []])
    self.assertEqual(swapped_alignment.cepts, [[], [], [2], [1]])