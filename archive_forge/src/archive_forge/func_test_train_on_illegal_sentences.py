import unittest
import pytest
from nltk import FreqDist
from nltk.lm import NgramCounter
from nltk.util import everygrams
def test_train_on_illegal_sentences(self):
    str_sent = ['Check', 'this', 'out', '!']
    list_sent = [['Check', 'this'], ['this', 'out'], ['out', '!']]
    with pytest.raises(TypeError):
        NgramCounter([str_sent])
    with pytest.raises(TypeError):
        NgramCounter([list_sent])