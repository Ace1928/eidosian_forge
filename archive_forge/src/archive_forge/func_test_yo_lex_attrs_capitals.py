import pytest
from spacy.lang.yo.lex_attrs import like_num
@pytest.mark.parametrize('word', ['eji', 'ejila', 'ogun', 'aárùn'])
def test_yo_lex_attrs_capitals(word):
    assert like_num(word)
    assert like_num(word.upper())