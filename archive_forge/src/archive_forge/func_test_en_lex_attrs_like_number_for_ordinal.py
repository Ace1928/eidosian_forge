import pytest
from spacy.lang.en.lex_attrs import like_num
@pytest.mark.parametrize('word', ['third', 'Millionth', '100th', 'Hundredth', '23rd', '52nd'])
def test_en_lex_attrs_like_number_for_ordinal(word):
    assert like_num(word)