import pytest
from nltk.tag import hmm
def test_forward_probability():
    from numpy.testing import assert_array_almost_equal
    model, states, symbols = hmm._market_hmm_example()
    seq = [('up', None), ('up', None)]
    expected = [[0.35, 0.02, 0.09], [0.1792, 0.0085, 0.0357]]
    fp = 2 ** model._forward_probability(seq)
    assert_array_almost_equal(fp, expected)