import numpy
import pytest
from spacy import registry, util
from spacy.lang.en import English
from spacy.pipeline import AttributeRuler
from spacy.tokens import Doc
from spacy.training import Example
from ..util import make_tempdir
def check_tag_map(ruler):
    doc = Doc(ruler.vocab, words=['This', 'is', 'a', 'test', '.'], tags=['DT', 'VBZ', 'DT', 'NN', '.'])
    doc = ruler(doc)
    for i in range(len(doc)):
        if i == 4:
            assert doc[i].pos_ == 'PUNCT'
            assert str(doc[i].morph) == 'PunctType=peri'
        else:
            assert doc[i].pos_ == ''
            assert str(doc[i].morph) == ''