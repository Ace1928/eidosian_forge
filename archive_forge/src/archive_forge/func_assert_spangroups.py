import pytest
from spacy.tokens import Span, SpanGroup
from spacy.tokens._dict_proxies import SpanGroups
def assert_spangroups():
    assert len(doc.spans) == 2
    assert doc.spans['test'].name == 'test'
    assert doc.spans['test2'].name == 'test'
    assert list(doc.spans['test']) == [doc[0:1]]
    assert list(doc.spans['test2']) == [doc[1:2]]