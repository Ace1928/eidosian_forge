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
def test_span_as_doc(doc):
    span = doc[4:10]
    span_doc = span.as_doc()
    assert span.text == span_doc.text.strip()
    assert isinstance(span_doc, doc.__class__)
    assert span_doc is not doc
    assert span_doc[0].idx == 0
    assert len(span_doc.ents) == 0
    span_doc = doc[2:10].as_doc()
    assert len(span_doc.ents) == 1
    span_doc = doc[0:5].as_doc()
    assert len(span_doc.ents) == 0