import unittest
from collections import defaultdict
from nltk.translate import AlignedSent, IBMModel, IBMModel2
from nltk.translate.ibm_model import AlignmentInfo
def test_set_uniform_alignment_probabilities(self):
    corpus = [AlignedSent(['ham', 'eggs'], ['schinken', 'schinken', 'eier']), AlignedSent(['spam', 'spam', 'spam', 'spam'], ['spam', 'spam'])]
    model2 = IBMModel2(corpus, 0)
    model2.set_uniform_probabilities(corpus)
    self.assertEqual(model2.alignment_table[0][1][3][2], 1.0 / 4)
    self.assertEqual(model2.alignment_table[2][4][2][4], 1.0 / 3)