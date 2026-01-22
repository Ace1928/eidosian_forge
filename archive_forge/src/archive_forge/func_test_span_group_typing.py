from random import Random
from typing import List
import pytest
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, SpanGroup
from spacy.util import filter_spans
@pytest.mark.issue(11975)
def test_span_group_typing(doc: Doc):
    """Tests whether typing of `SpanGroup` as `Iterable[Span]`-like object is accepted by mypy."""
    span_group: SpanGroup = doc.spans['SPANS']
    spans: List[Span] = list(span_group)
    for i, span in enumerate(span_group):
        assert span == span_group[i] == spans[i]
    filter_spans(span_group)