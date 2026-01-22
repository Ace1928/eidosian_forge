import logging
import unittest
import numpy as np
from gensim.models.phrases import Phrases, FrozenPhrases, _PhrasesTransformation
from gensim.models.phrases import original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath
def test_sentence_generation_with_generator(self):
    """Test basic bigram production when corpus is a generator."""
    self.assertEqual(len(list(self.gen_sentences())), len(list(self.bigram_default[self.gen_sentences()])))