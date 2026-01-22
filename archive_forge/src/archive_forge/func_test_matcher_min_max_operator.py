import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_matcher_min_max_operator(en_vocab):
    doc = Doc(en_vocab, words=['foo', 'bar', 'foo', 'foo', 'bar', 'foo', 'foo', 'foo', 'bar', 'bar'])
    matcher = Matcher(en_vocab)
    pattern = [{'ORTH': 'foo', 'OP': '{3}'}]
    matcher.add('TEST', [pattern])
    matches1 = [doc[start:end].text for _, start, end in matcher(doc)]
    assert len(matches1) == 1
    matcher = Matcher(en_vocab)
    pattern = [{'ORTH': 'foo', 'OP': '{2,}'}]
    matcher.add('TEST', [pattern])
    matches2 = [doc[start:end].text for _, start, end in matcher(doc)]
    assert len(matches2) == 4
    matcher = Matcher(en_vocab)
    pattern = [{'ORTH': 'foo', 'OP': '{,2}'}]
    matcher.add('TEST', [pattern])
    matches3 = [doc[start:end].text for _, start, end in matcher(doc)]
    assert len(matches3) == 9
    matcher = Matcher(en_vocab)
    pattern = [{'ORTH': 'foo', 'OP': '{2,3}'}]
    matcher.add('TEST', [pattern])
    matches4 = [doc[start:end].text for _, start, end in matcher(doc)]
    assert len(matches4) == 4