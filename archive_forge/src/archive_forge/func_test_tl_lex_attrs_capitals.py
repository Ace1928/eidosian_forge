import pytest
from spacy.lang.tl.lex_attrs import like_num
@pytest.mark.xfail(reason='Not yet implemented, fails when capitalized.')
@pytest.mark.parametrize('word', ['isa', 'dalawa', 'tatlo'])
def test_tl_lex_attrs_capitals(word):
    assert like_num(word)
    assert like_num(word.upper())