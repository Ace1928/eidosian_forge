import numpy
import pytest
from spacy.attrs import IS_ALPHA, IS_DIGIT
from spacy.lookups import Lookups
from spacy.tokens import Doc
from spacy.util import OOV_RANK
from spacy.vocab import Vocab
def test_vocab_lexeme_is_digit(en_vocab):
    assert not en_vocab['the'].flags & 1 << IS_DIGIT
    assert en_vocab['1999'].flags & 1 << IS_DIGIT
    assert not en_vocab['hello1'].flags & 1 << IS_DIGIT