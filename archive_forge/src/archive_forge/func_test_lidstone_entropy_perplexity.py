import math
from operator import itemgetter
import pytest
from nltk.lm import (
from nltk.lm.preprocessing import padded_everygrams
def test_lidstone_entropy_perplexity(lidstone_bigram_model):
    text = [('<s>', 'a'), ('a', 'c'), ('c', '<UNK>'), ('<UNK>', 'd'), ('d', 'c'), ('c', '</s>')]
    H = 4.0917
    perplexity = 17.0504
    assert pytest.approx(lidstone_bigram_model.entropy(text), 0.0001) == H
    assert pytest.approx(lidstone_bigram_model.perplexity(text), 0.0001) == perplexity