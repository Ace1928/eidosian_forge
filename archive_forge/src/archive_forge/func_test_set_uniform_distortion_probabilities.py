import unittest
from collections import defaultdict
from nltk.translate import AlignedSent, IBMModel, IBMModel3
from nltk.translate.ibm_model import AlignmentInfo
def test_set_uniform_distortion_probabilities(self):
    corpus = [AlignedSent(['ham', 'eggs'], ['schinken', 'schinken', 'eier']), AlignedSent(['spam', 'spam', 'spam', 'spam'], ['spam', 'spam'])]
    model3 = IBMModel3(corpus, 0)
    model3.set_uniform_probabilities(corpus)
    self.assertEqual(model3.distortion_table[1][0][3][2], 1.0 / 2)
    self.assertEqual(model3.distortion_table[4][2][2][4], 1.0 / 4)