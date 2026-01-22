import logging
import unittest
import numpy as np
from gensim.models.phrases import Phrases, FrozenPhrases, _PhrasesTransformation
from gensim.models.phrases import original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath
def test_simple_analysis(self):
    """Test transformation with no phrases."""
    sentence = ['simple', 'sentence', 'should', 'pass']
    result = self.AnalysisTester({}, threshold=1)[sentence]
    self.assertEqual(result, sentence)
    sentence = ['a', 'simple', 'sentence', 'with', 'no', 'bigram', 'but', 'common', 'terms']
    result = self.AnalysisTester({}, threshold=1)[sentence]
    self.assertEqual(result, sentence)