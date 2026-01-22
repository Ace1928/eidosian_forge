import numpy
import pytest
from spacy.attrs import IS_ALPHA, IS_DIGIT, IS_LOWER, IS_PUNCT, IS_STOP, IS_TITLE
from spacy.symbols import VERB
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
def test_set_invalid_pos():
    doc = Doc(Vocab(), words=['hello', 'world'])
    with pytest.raises(ValueError):
        doc[0].pos_ = 'blah'