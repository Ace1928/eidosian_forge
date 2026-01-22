import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_matcher_match_zero(matcher):
    words1 = 'He said , " some words " ...'.split()
    words2 = 'He said , " some three words " ...'.split()
    pattern1 = [{'ORTH': '"'}, {'OP': '!', 'IS_PUNCT': True}, {'OP': '!', 'IS_PUNCT': True}, {'ORTH': '"'}]
    pattern2 = [{'ORTH': '"'}, {'IS_PUNCT': True}, {'IS_PUNCT': True}, {'IS_PUNCT': True}, {'ORTH': '"'}]
    matcher.add('Quote', [pattern1])
    doc = Doc(matcher.vocab, words=words1)
    assert len(matcher(doc)) == 1
    doc = Doc(matcher.vocab, words=words2)
    assert len(matcher(doc)) == 0
    matcher.add('Quote', [pattern2])
    assert len(matcher(doc)) == 0