import pytest
from spacy.attrs import ENT_IOB, IS_ALPHA, LEMMA, NORM, ORTH, intify_attrs
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.lex_attrs import (
def test_attrs_ent_iob_intify():
    int_attrs = intify_attrs({'ENT_IOB': ''})
    assert int_attrs == {ENT_IOB: 0}
    int_attrs = intify_attrs({'ENT_IOB': 'I'})
    assert int_attrs == {ENT_IOB: 1}
    int_attrs = intify_attrs({'ENT_IOB': 'O'})
    assert int_attrs == {ENT_IOB: 2}
    int_attrs = intify_attrs({'ENT_IOB': 'B'})
    assert int_attrs == {ENT_IOB: 3}
    int_attrs = intify_attrs({ENT_IOB: ''})
    assert int_attrs == {ENT_IOB: 0}
    int_attrs = intify_attrs({ENT_IOB: 'I'})
    assert int_attrs == {ENT_IOB: 1}
    int_attrs = intify_attrs({ENT_IOB: 'O'})
    assert int_attrs == {ENT_IOB: 2}
    int_attrs = intify_attrs({ENT_IOB: 'B'})
    assert int_attrs == {ENT_IOB: 3}
    with pytest.raises(ValueError):
        int_attrs = intify_attrs({'ENT_IOB': 'XX'})
    with pytest.raises(ValueError):
        int_attrs = intify_attrs({ENT_IOB: 'XX'})