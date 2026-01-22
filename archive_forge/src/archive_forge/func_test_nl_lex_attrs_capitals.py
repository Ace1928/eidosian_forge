import pytest
from spacy.lang.nl.lex_attrs import like_num
@pytest.mark.parametrize('word', ['elf', 'elfde'])
def test_nl_lex_attrs_capitals(word):
    assert like_num(word)
    assert like_num(word.upper())