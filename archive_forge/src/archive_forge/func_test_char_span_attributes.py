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
def test_char_span_attributes(doc):
    label = 'LABEL'
    kb_id = 'KB_ID'
    span_id = 'SPAN_ID'
    span1 = doc.char_span(20, 45, label=label, kb_id=kb_id, span_id=span_id)
    span2 = doc[1:].char_span(15, 40, label=label, kb_id=kb_id, span_id=span_id)
    assert span1.text == span2.text
    assert span1.label_ == span2.label_ == label
    assert span1.kb_id_ == span2.kb_id_ == kb_id
    assert span1.id_ == span2.id_ == span_id