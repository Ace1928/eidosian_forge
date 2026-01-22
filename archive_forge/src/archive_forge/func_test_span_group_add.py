from random import Random
from typing import List
import pytest
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, SpanGroup
from spacy.util import filter_spans
def test_span_group_add(doc):
    span_group_1 = doc.spans['SPANS']
    spans = [doc[0:5], doc[0:6]]
    span_group_2 = SpanGroup(doc, name='MORE_SPANS', attrs={'key': 'new_value', 'new_key': 'new_value'}, spans=spans)
    span_group_3_expected = span_group_1._concat(span_group_2)
    span_group_3 = span_group_1 + span_group_2
    assert len(span_group_3) == len(span_group_3_expected)
    assert span_group_3.attrs == {'key': 'value', 'new_key': 'new_value'}
    assert list(span_group_3) == list(span_group_3_expected)