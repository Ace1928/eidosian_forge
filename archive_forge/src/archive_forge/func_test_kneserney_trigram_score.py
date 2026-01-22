import math
from operator import itemgetter
import pytest
from nltk.lm import (
from nltk.lm.preprocessing import padded_everygrams
@pytest.mark.parametrize('word, context, expected_score', [('c', None, 1.0 / 14), ('z', None, 0.0 / 14), ('y', None, 3 / 14), ('c', ['b'], 0.125 + 0.75 * (1 / 14)), ('c', ['a', 'b'], 0.25 + 0.75 * (0.125 + 0.75 * (1 / 14))), ('c', ['z', 'b'], 0.125 + 0.75 * (1 / 14))])
def test_kneserney_trigram_score(kneserney_trigram_model, word, context, expected_score):
    assert pytest.approx(kneserney_trigram_model.score(word, context), 0.0001) == expected_score