import logging
import unittest
import numpy as np
from gensim.models.phrases import Phrases, FrozenPhrases, _PhrasesTransformation
from gensim.models.phrases import original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath
def test_analysis_connector_words_in_between(self):
    scores = {'simple_sentence': 2, 'sentence_with_many': 2, 'many_possible': 2, 'many_of_the_possible': 2, 'possible_bigrams': 2}
    sentence = ['sentence', 'with', 'many', 'possible', 'bigrams']
    result = self.AnalysisTester(scores, threshold=1)[sentence]
    self.assertEqual(result, ['sentence_with_many', 'possible_bigrams'])
    sentence = ['a', 'simple', 'sentence', 'with', 'many', 'of', 'the', 'possible', 'bigrams', 'with']
    result = self.AnalysisTester(scores, threshold=1)[sentence]
    self.assertEqual(result, ['a', 'simple_sentence', 'with', 'many_of_the_possible', 'bigrams', 'with'])