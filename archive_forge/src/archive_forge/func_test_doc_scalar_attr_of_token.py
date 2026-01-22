import numpy
import pytest
from spacy.attrs import DEP, MORPH, ORTH, POS, SHAPE
from spacy.tokens import Doc
def test_doc_scalar_attr_of_token(en_vocab):
    doc = Doc(en_vocab, words=['An', 'example', 'sentence'])
    example = doc.vocab['example']
    assert example.orth != example.shape
    feats_array = doc.to_array(ORTH)
    assert feats_array.shape == (3,)