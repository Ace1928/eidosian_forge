import numpy
import pytest
from spacy.attrs import IS_ALPHA, IS_DIGIT, IS_LOWER, IS_PUNCT, IS_STOP, IS_TITLE
from spacy.symbols import VERB
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
def test_doc_token_api_vectors():
    vocab = Vocab()
    vocab.reset_vectors(width=2)
    vocab.set_vector('apples', vector=numpy.asarray([0.0, 2.0], dtype='f'))
    vocab.set_vector('oranges', vector=numpy.asarray([0.0, 1.0], dtype='f'))
    doc = Doc(vocab, words=['apples', 'oranges', 'oov'])
    assert doc.has_vector
    assert doc[0].has_vector
    assert doc[1].has_vector
    assert not doc[2].has_vector
    apples_norm = (0 * 0 + 2 * 2) ** 0.5
    oranges_norm = (0 * 0 + 1 * 1) ** 0.5
    cosine = (0 * 0 + 2 * 1) / (apples_norm * oranges_norm)
    assert doc[0].similarity(doc[1]) == cosine