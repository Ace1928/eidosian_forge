import unittest
from collections import Counter
from timeit import timeit
from nltk.lm import Vocabulary
@unittest.skip(reason='Test is known to be flaky as it compares (runtime) performance.')
def test_len_is_constant(self):
    small_vocab = Vocabulary('abcde')
    from nltk.corpus.europarl_raw import english
    large_vocab = Vocabulary(english.words())
    small_vocab_len_time = timeit('len(small_vocab)', globals=locals())
    large_vocab_len_time = timeit('len(large_vocab)', globals=locals())
    self.assertAlmostEqual(small_vocab_len_time, large_vocab_len_time, places=1)