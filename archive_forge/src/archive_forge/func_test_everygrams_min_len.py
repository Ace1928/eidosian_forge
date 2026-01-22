import pytest
from nltk.util import everygrams
def test_everygrams_min_len(everygram_input):
    expected_output = [('a', 'b'), ('a', 'b', 'c'), ('b', 'c')]
    output = list(everygrams(everygram_input, min_len=2))
    assert output == expected_output