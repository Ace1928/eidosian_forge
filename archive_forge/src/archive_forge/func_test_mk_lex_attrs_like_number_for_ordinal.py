import pytest
from spacy.lang.mk.lex_attrs import like_num
@pytest.mark.parametrize('word', ['првиот', 'втора', 'четврт', 'четвртата', 'петти', 'петто', 'стоти', 'шеесетите', 'седумдесетите'])
def test_mk_lex_attrs_like_number_for_ordinal(word):
    assert like_num(word)