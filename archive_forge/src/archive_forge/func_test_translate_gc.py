from collections import namedtuple
import unittest
import logging
import numpy as np
import pytest
from scipy.spatial.distance import cosine
from gensim.models.doc2vec import Doc2Vec
from gensim import utils
from gensim.models import translation_matrix
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
@pytest.mark.xfail(True, reason='blinking test, can be related to <https://github.com/RaRe-Technologies/gensim/issues/2977>')
def test_translate_gc(self):
    model = translation_matrix.TranslationMatrix(self.source_word_vec, self.target_word_vec, self.word_pairs)
    model.train(self.word_pairs)
    test_source_word, test_target_word = zip(*self.test_word_pairs)
    translated_words = model.translate(test_source_word, topn=5, gc=1, sample_num=3, source_lang_vec=self.source_word_vec, target_lang_vec=self.target_word_vec)
    for idx, item in enumerate(self.test_word_pairs):
        self.assertTrue(item[1] in translated_words[item[0]])