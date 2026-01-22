import pytest
from spacy.tokens import Span, SpanGroup
from spacy.tokens._dict_proxies import SpanGroups
@pytest.mark.issue(10685)
def test_issue10685(en_tokenizer):
    """Test `SpanGroups` de/serialization"""
    doc = en_tokenizer('Will it blend?')
    assert len(doc.spans) == 0
    doc.spans.from_bytes(doc.spans.to_bytes())
    assert len(doc.spans) == 0
    doc.spans['test'] = SpanGroup(doc, name='test', spans=[doc[0:1]])
    doc.spans['test2'] = SpanGroup(doc, name='test', spans=[doc[1:2]])

    def assert_spangroups():
        assert len(doc.spans) == 2
        assert doc.spans['test'].name == 'test'
        assert doc.spans['test2'].name == 'test'
        assert list(doc.spans['test']) == [doc[0:1]]
        assert list(doc.spans['test2']) == [doc[1:2]]
    assert_spangroups()
    doc.spans.from_bytes(doc.spans.to_bytes())
    assert_spangroups()