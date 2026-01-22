import math
from operator import itemgetter
import pytest
from nltk.lm import (
from nltk.lm.preprocessing import padded_everygrams
@pytest.mark.parametrize('word, context, expected_score', [('c', None, 1.0 / 18), ('z', None, 0.0 / 18), ('y', None, 3 / 18), ('c', ['b'], 0.125 + 0.75 * (2 / 2) * (1 / 18)), ('c', ['a', 'b'], 0.25 + 0.75 * (0.125 + 0.75 * (2 / 2) * (1 / 18))), ('c', ['z', 'b'], 0.125 + 0.75 * (2 / 2) * (1 / 18))])
def test_absolute_discounting_trigram_score(absolute_discounting_trigram_model, word, context, expected_score):
    assert pytest.approx(absolute_discounting_trigram_model.score(word, context), 0.0001) == expected_score