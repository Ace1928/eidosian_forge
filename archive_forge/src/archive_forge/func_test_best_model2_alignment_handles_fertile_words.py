import unittest
from collections import defaultdict
from nltk.translate import AlignedSent, IBMModel
from nltk.translate.ibm_model import AlignmentInfo
def test_best_model2_alignment_handles_fertile_words(self):
    sentence_pair = AlignedSent(['i', 'really', ',', 'really', 'love', 'ham'], TestIBMModel.__TEST_SRC_SENTENCE)
    translation_table = {'i': {"j'": 0.9, 'aime': 0.05, 'bien': 0.02, 'jambon': 0.03, None: 0}, 'really': {"j'": 0, 'aime': 0, 'bien': 0.9, 'jambon': 0.01, None: 0.09}, ',': {"j'": 0, 'aime': 0, 'bien': 0.3, 'jambon': 0, None: 0.7}, 'love': {"j'": 0.05, 'aime': 0.9, 'bien': 0.01, 'jambon': 0.01, None: 0.03}, 'ham': {"j'": 0, 'aime': 0.01, 'bien': 0, 'jambon': 0.99, None: 0}}
    alignment_table = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.2))))
    ibm_model = IBMModel([])
    ibm_model.translation_table = translation_table
    ibm_model.alignment_table = alignment_table
    a_info = ibm_model.best_model2_alignment(sentence_pair)
    self.assertEqual(a_info.alignment[1:], (1, 3, 0, 3, 2, 4))
    self.assertEqual(a_info.cepts, [[3], [1], [5], [2, 4], [6]])