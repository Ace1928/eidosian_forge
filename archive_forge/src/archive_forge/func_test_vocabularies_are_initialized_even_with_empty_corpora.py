import unittest
from collections import defaultdict
from nltk.translate import AlignedSent, IBMModel
from nltk.translate.ibm_model import AlignmentInfo
def test_vocabularies_are_initialized_even_with_empty_corpora(self):
    parallel_corpora = []
    ibm_model = IBMModel(parallel_corpora)
    self.assertEqual(len(ibm_model.src_vocab), 1)
    self.assertEqual(len(ibm_model.trg_vocab), 0)