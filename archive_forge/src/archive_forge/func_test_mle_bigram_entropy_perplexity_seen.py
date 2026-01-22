import math
from operator import itemgetter
import pytest
from nltk.lm import (
from nltk.lm.preprocessing import padded_everygrams
def test_mle_bigram_entropy_perplexity_seen(mle_bigram_model):
    trained = [('<s>', 'a'), ('a', 'b'), ('b', '<UNK>'), ('<UNK>', 'a'), ('a', 'd'), ('d', '</s>')]
    H = 1.0975
    perplexity = 2.1398
    assert pytest.approx(mle_bigram_model.entropy(trained), 0.0001) == H
    assert pytest.approx(mle_bigram_model.perplexity(trained), 0.0001) == perplexity