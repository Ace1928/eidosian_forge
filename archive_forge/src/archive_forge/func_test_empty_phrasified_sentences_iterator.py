import logging
import unittest
import numpy as np
from gensim.models.phrases import Phrases, FrozenPhrases, _PhrasesTransformation
from gensim.models.phrases import original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath
def test_empty_phrasified_sentences_iterator(self):
    bigram_phrases = Phrases(self.sentences)
    bigram_phraser = FrozenPhrases(bigram_phrases)
    trigram_phrases = Phrases(bigram_phraser[self.sentences])
    trigram_phraser = FrozenPhrases(trigram_phrases)
    trigrams = trigram_phraser[bigram_phraser[self.sentences]]
    fst, snd = (list(trigrams), list(trigrams))
    self.assertEqual(fst, snd)
    self.assertNotEqual(snd, [])