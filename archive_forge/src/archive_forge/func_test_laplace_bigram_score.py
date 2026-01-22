import math
from operator import itemgetter
import pytest
from nltk.lm import (
from nltk.lm.preprocessing import padded_everygrams
@pytest.mark.parametrize('word, context, expected_score', [('d', ['c'], 2.0 / 9), ('a', None, 3.0 / 22), ('z', None, 1.0 / 22), ('y', None, 4.0 / 22)])
def test_laplace_bigram_score(laplace_bigram_model, word, context, expected_score):
    assert pytest.approx(laplace_bigram_model.score(word, context), 0.0001) == expected_score