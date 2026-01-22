import math
from operator import itemgetter
import pytest
from nltk.lm import (
from nltk.lm.preprocessing import padded_everygrams
@pytest.mark.parametrize('word, context, expected_score', [('d', ('b', 'c'), 1), ('d', ['c'], 1), ('a', None, 2.0 / 18), ('z', None, 0), ('y', None, 3.0 / 18)])
def test_mle_trigram_scores(mle_trigram_model, word, context, expected_score):
    assert pytest.approx(mle_trigram_model.score(word, context), 0.0001) == expected_score