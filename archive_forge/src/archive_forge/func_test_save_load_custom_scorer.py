import logging
import unittest
import numpy as np
from gensim.models.phrases import Phrases, FrozenPhrases, _PhrasesTransformation
from gensim.models.phrases import original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath
def test_save_load_custom_scorer(self):
    """Test saving and loading a FrozenPhrases object with a custom scorer."""
    with temporary_file('test.pkl') as fpath:
        bigram = FrozenPhrases(Phrases(self.sentences, min_count=1, threshold=0.001, scoring=dumb_scorer))
        bigram.save(fpath)
        bigram_loaded = FrozenPhrases.load(fpath)
        self.assertEqual(bigram_loaded.scoring, dumb_scorer)