import logging
import unittest
import numpy as np
from gensim.models.phrases import Phrases, FrozenPhrases, _PhrasesTransformation
from gensim.models.phrases import original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath
def test_save_load_no_scoring(self):
    """Test saving and loading a FrozenPhrases object with no scoring parameter.
        This should ensure backwards compatibility with old versions of FrozenPhrases"""
    bigram_loaded = FrozenPhrases.load(datapath('phraser-no-scoring.pkl'))
    self.assertEqual(bigram_loaded.scoring, original_scorer)