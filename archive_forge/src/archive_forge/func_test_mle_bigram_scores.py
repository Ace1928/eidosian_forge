import math
from operator import itemgetter
import pytest
from nltk.lm import (
from nltk.lm.preprocessing import padded_everygrams
@pytest.mark.parametrize('word, context, expected_score', [('d', ['c'], 1), ('d', ['e'], 0), ('z', None, 0), ('a', None, 2.0 / 14), ('y', None, 3.0 / 14)])
def test_mle_bigram_scores(mle_bigram_model, word, context, expected_score):
    assert pytest.approx(mle_bigram_model.score(word, context), 0.0001) == expected_score