from random import Random
from typing import List
import pytest
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, SpanGroup
from spacy.util import filter_spans
def test_span_doc_delitem(doc):
    span_group = doc.spans['SPANS']
    length = len(span_group)
    index = 5
    span = span_group[index]
    next_span = span_group[index + 1]
    del span_group[index]
    assert len(span_group) == length - 1
    assert span_group[index] != span
    assert span_group[index] == next_span
    with pytest.raises(IndexError):
        del span_group[-100]
    with pytest.raises(IndexError):
        del span_group[100]