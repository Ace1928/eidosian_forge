import numpy
import pytest
from spacy.attrs import DEP, MORPH, ORTH, POS, SHAPE
from spacy.tokens import Doc
def test_doc_array_tag(en_vocab):
    words = ['A', 'nice', 'sentence', '.']
    pos = ['DET', 'ADJ', 'NOUN', 'PUNCT']
    doc = Doc(en_vocab, words=words, pos=pos)
    assert doc[0].pos != doc[1].pos != doc[2].pos != doc[3].pos
    feats_array = doc.to_array((ORTH, POS))
    assert feats_array[0][1] == doc[0].pos
    assert feats_array[1][1] == doc[1].pos
    assert feats_array[2][1] == doc[2].pos
    assert feats_array[3][1] == doc[3].pos