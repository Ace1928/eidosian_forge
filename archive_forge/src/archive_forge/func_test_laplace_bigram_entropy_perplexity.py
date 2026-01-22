import math
from operator import itemgetter
import pytest
from nltk.lm import (
from nltk.lm.preprocessing import padded_everygrams
def test_laplace_bigram_entropy_perplexity(laplace_bigram_model):
    text = [('<s>', 'a'), ('a', 'c'), ('c', '<UNK>'), ('<UNK>', 'd'), ('d', 'c'), ('c', '</s>')]
    H = 3.1275
    perplexity = 8.7393
    assert pytest.approx(laplace_bigram_model.entropy(text), 0.0001) == H
    assert pytest.approx(laplace_bigram_model.perplexity(text), 0.0001) == perplexity