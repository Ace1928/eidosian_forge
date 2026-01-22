import pytest
from spacy.attrs import ENT_IOB, IS_ALPHA, LEMMA, NORM, ORTH, intify_attrs
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.lex_attrs import (
@pytest.mark.parametrize('text', ['dog'])
def test_attrs_do_deprecated(text):
    int_attrs = intify_attrs({'F': text, 'is_alpha': True}, strings_map={text: 10}, _do_deprecated=True)
    assert int_attrs == {ORTH: 10, IS_ALPHA: True}