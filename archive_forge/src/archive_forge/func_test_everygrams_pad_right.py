import pytest
from nltk.util import everygrams
def test_everygrams_pad_right(everygram_input):
    expected_output = [('a',), ('a', 'b'), ('a', 'b', 'c'), ('b',), ('b', 'c'), ('b', 'c', None), ('c',), ('c', None), ('c', None, None), (None,), (None, None), (None,)]
    output = list(everygrams(everygram_input, max_len=3, pad_right=True))
    assert output == expected_output