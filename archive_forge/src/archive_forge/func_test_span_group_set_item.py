from random import Random
from typing import List
import pytest
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, SpanGroup
from spacy.util import filter_spans
def test_span_group_set_item(doc, other_doc):
    span_group = doc.spans['SPANS']
    index = 5
    span = span_group[index]
    span.label_ = 'NEW LABEL'
    span.kb_id = doc.vocab.strings['KB_ID']
    assert span_group[index].label != span.label
    assert span_group[index].kb_id != span.kb_id
    span_group[index] = span
    assert span_group[index].start == span.start
    assert span_group[index].end == span.end
    assert span_group[index].label == span.label
    assert span_group[index].kb_id == span.kb_id
    assert span_group[index] == span
    with pytest.raises(IndexError):
        span_group[-100] = span
    with pytest.raises(IndexError):
        span_group[100] = span
    span = Span(other_doc, 0, 2)
    with pytest.raises(ValueError):
        span_group[index] = span