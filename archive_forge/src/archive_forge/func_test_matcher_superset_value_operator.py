import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_matcher_superset_value_operator(en_vocab):
    matcher = Matcher(en_vocab)
    pattern = [{'MORPH': {'IS_SUPERSET': ['Feat=Val', 'Feat2=Val2', 'Feat3=Val3']}}]
    matcher.add('M', [pattern])
    doc = Doc(en_vocab, words=['a', 'b', 'c'])
    assert len(matcher(doc)) == 0
    doc[0].set_morph('Feat=Val|Feat2=Val2')
    assert len(matcher(doc)) == 0
    doc[0].set_morph('Feat=Val|Feat2=Val2|Feat3=Val3')
    assert len(matcher(doc)) == 1
    doc[0].set_morph('Feat=Val|Feat2=Val2|Feat3=Val3|Feat4=Val4')
    assert len(matcher(doc)) == 1
    matcher = Matcher(en_vocab)
    pattern = [{'TAG': {'IS_SUPERSET': ['A', 'B']}}]
    matcher.add('M', [pattern])
    doc = Doc(en_vocab, words=['a', 'b', 'c'])
    doc[0].tag_ = 'A'
    assert len(matcher(doc)) == 0
    matcher = Matcher(en_vocab)
    pattern = [{'TAG': {'IS_SUPERSET': ['A']}}]
    matcher.add('M', [pattern])
    doc = Doc(en_vocab, words=['a', 'b', 'c'])
    doc[0].tag_ = 'A'
    assert len(matcher(doc)) == 1
    matcher = Matcher(en_vocab)
    pattern = [{'TAG': {'IS_SUPERSET': []}}]
    matcher.add('M', [pattern])
    doc = Doc(en_vocab, words=['a', 'b', 'c'])
    doc[0].tag_ = 'A'
    assert len(matcher(doc)) == 3
    Token.set_extension('ext', default=[])
    matcher = Matcher(en_vocab)
    pattern = [{'_': {'ext': {'IS_SUPERSET': ['A']}}}]
    matcher.add('M', [pattern])
    doc = Doc(en_vocab, words=['a', 'b', 'c'])
    doc[0]._.ext = ['A', 'B']
    assert len(matcher(doc)) == 1