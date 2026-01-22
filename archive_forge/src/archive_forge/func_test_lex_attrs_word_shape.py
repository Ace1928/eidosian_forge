import pytest
from spacy.attrs import ENT_IOB, IS_ALPHA, LEMMA, NORM, ORTH, intify_attrs
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.lex_attrs import (
@pytest.mark.parametrize('text,shape', [('Nasa', 'Xxxx'), ('capitalized', 'xxxx'), ('999999999', 'dddd'), ('C3P0', 'XdXd'), (',', ','), ('\n', '\n'), ('``,-', '``,-')])
def test_lex_attrs_word_shape(text, shape):
    assert word_shape(text) == shape