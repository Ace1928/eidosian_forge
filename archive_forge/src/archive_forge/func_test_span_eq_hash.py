import numpy
import pytest
from numpy.testing import assert_array_equal
from thinc.api import get_current_ops
from spacy.attrs import LENGTH, ORTH
from spacy.lang.en import English
from spacy.tokens import Doc, Span, Token
from spacy.util import filter_spans
from spacy.vocab import Vocab
from ..util import add_vecs_to_vocab
from .test_underscore import clean_underscore  # noqa: F401
def test_span_eq_hash(doc, doc_not_parsed):
    assert doc[0:2] == doc[0:2]
    assert doc[0:2] != doc[1:3]
    assert doc[0:2] != doc_not_parsed[0:2]
    assert hash(doc[0:2]) == hash(doc[0:2])
    assert hash(doc[0:2]) != hash(doc[1:3])
    assert hash(doc[0:2]) != hash(doc_not_parsed[0:2])
    assert doc[0:len(doc)] != doc[len(doc):len(doc) + 1]