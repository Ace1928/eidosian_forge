import logging
import unittest
import numpy as np
from gensim.models.phrases import Phrases, FrozenPhrases, _PhrasesTransformation
from gensim.models.phrases import original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath
def test_analysis_connector_words(self):
    scores = {'simple_sentence': 2, 'sentence_many': 2, 'many_possible': 2, 'possible_bigrams': 2}
    sentence = ['a', 'simple', 'sentence', 'many', 'the', 'possible', 'bigrams']
    result = self.AnalysisTester(scores, threshold=1)[sentence]
    self.assertEqual(result, ['a', 'simple_sentence', 'many', 'the', 'possible_bigrams'])
    sentence = ['simple', 'the', 'sentence', 'and', 'many', 'possible', 'bigrams', 'with', 'a']
    result = self.AnalysisTester(scores, threshold=1)[sentence]
    self.assertEqual(result, ['simple', 'the', 'sentence', 'and', 'many_possible', 'bigrams', 'with', 'a'])