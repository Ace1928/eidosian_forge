import pytest
from spacy.tokens import Span, SpanGroup
from spacy.tokens._dict_proxies import SpanGroups
def test_span_groups_serialization_mismatches(en_tokenizer):
    """Test the serialization of multiple mismatching `SpanGroups` keys and `SpanGroup.name`s"""
    doc = en_tokenizer('How now, brown cow?')
    groups = doc.spans
    groups['key1'] = SpanGroup(doc, name='key1', spans=[doc[0:1], doc[1:2]])
    groups['key2'] = SpanGroup(doc, name='too', spans=[doc[3:4], doc[4:5]])
    groups['key3'] = SpanGroup(doc, name='too', spans=[doc[1:2], doc[0:1]])
    groups['key4'] = SpanGroup(doc, name='key4', spans=[doc[0:1]])
    groups['key5'] = SpanGroup(doc, name='key4', spans=[doc[0:1]])
    sg6 = SpanGroup(doc, name='key6', spans=[doc[0:1]])
    groups['key6'] = sg6
    groups['key7'] = sg6
    sg8 = SpanGroup(doc, name='also', spans=[doc[1:2]])
    groups['key8'] = sg8
    groups['key9'] = sg8
    regroups = SpanGroups(doc).from_bytes(groups.to_bytes())
    assert regroups.keys() == groups.keys()
    for key, regroup in regroups.items():
        assert regroup.name == groups[key].name
        assert list(regroup) == list(groups[key])