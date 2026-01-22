import numpy
import pytest
from spacy import registry, util
from spacy.lang.en import English
from spacy.pipeline import AttributeRuler
from spacy.tokens import Doc
from spacy.training import Example
from ..util import make_tempdir
def check_morph_rules(ruler):
    doc = Doc(ruler.vocab, words=['This', 'is', 'the', 'test', '.'], tags=['DT', 'VBZ', 'DT', 'NN', '.'])
    doc = ruler(doc)
    for i in range(len(doc)):
        if i != 2:
            assert doc[i].pos_ == ''
            assert str(doc[i].morph) == ''
        else:
            assert doc[2].pos_ == 'DET'
            assert doc[2].lemma_ == 'a'
            assert str(doc[2].morph) == 'Case=Nom'