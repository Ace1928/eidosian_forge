import pytest
from spacy.attrs import ENT_IOB, IS_ALPHA, LEMMA, NORM, ORTH, intify_attrs
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.lex_attrs import (
@pytest.mark.parametrize('text,match', [('www.google.com', True), ('google.com', True), ('sydney.com', True), ('1abc2def.org', True), ('http://stupid', True), ('www.hi', True), ('example.com/example', True), ('dog', False), ('1.2', False), ('1.a', False), ('hello.There', False)])
def test_lex_attrs_like_url(text, match):
    assert like_url(text) == match