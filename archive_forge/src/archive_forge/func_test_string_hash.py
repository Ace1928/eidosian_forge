import pytest
from spacy.strings import StringStore
def test_string_hash(stringstore):
    """Test that string hashing is stable across platforms"""
    assert stringstore.add('apple') == 8566208034543834098
    heart = '💙'
    h = stringstore.add(heart)
    assert h == 11841826740069053588