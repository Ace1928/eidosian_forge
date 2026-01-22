import unittest
import pytest
from nltk import FreqDist
from nltk.lm import NgramCounter
from nltk.util import everygrams
def test_train_on_unigrams(self):
    words = list('abcd')
    counter = NgramCounter([[(w,) for w in words]])
    assert not counter[3]
    assert not counter[2]
    self.case.assertCountEqual(words, counter[1].keys())