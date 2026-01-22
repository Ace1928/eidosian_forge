import numpy
import pytest
from spacy.attrs import DEP, MORPH, ORTH, POS, SHAPE
from spacy.tokens import Doc
def test_doc_array_dep(en_vocab):
    words = ['A', 'nice', 'sentence', '.']
    deps = ['det', 'amod', 'ROOT', 'punct']
    doc = Doc(en_vocab, words=words, deps=deps)
    feats_array = doc.to_array((ORTH, DEP))
    assert feats_array[0][1] == doc[0].dep
    assert feats_array[1][1] == doc[1].dep
    assert feats_array[2][1] == doc[2].dep
    assert feats_array[3][1] == doc[3].dep