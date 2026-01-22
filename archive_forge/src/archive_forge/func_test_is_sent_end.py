import numpy
import pytest
from spacy.attrs import IS_ALPHA, IS_DIGIT, IS_LOWER, IS_PUNCT, IS_STOP, IS_TITLE
from spacy.symbols import VERB
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
def test_is_sent_end(en_tokenizer):
    doc = en_tokenizer('This is a sentence. This is another.')
    assert doc[4].is_sent_end is None
    doc[5].is_sent_start = True
    assert doc[4].is_sent_end is True
    assert len(list(doc.sents)) == 2