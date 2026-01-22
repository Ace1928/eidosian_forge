import numpy
import pytest
from spacy.attrs import IS_ALPHA, IS_DIGIT, IS_LOWER, IS_PUNCT, IS_STOP, IS_TITLE
from spacy.symbols import VERB
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
def test_doc_token_api_is_properties(en_vocab):
    doc = Doc(en_vocab, words=['Hi', ',', 'my', 'email', 'is', 'test@me.com'])
    assert doc[0].is_title
    assert doc[0].is_alpha
    assert not doc[0].is_digit
    assert doc[1].is_punct
    assert doc[3].is_ascii
    assert not doc[3].like_url
    assert doc[4].is_lower
    assert doc[5].like_email