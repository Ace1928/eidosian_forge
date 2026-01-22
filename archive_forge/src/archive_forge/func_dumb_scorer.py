import logging
import unittest
import numpy as np
from gensim.models.phrases import Phrases, FrozenPhrases, _PhrasesTransformation
from gensim.models.phrases import original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath
def dumb_scorer(worda_count, wordb_count, bigram_count, len_vocab, min_count, corpus_word_count):
    return 1