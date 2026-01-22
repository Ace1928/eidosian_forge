import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_matcher_match_zero_plus(matcher):
    words = 'He said , " some words " ...'.split()
    pattern = [{'ORTH': '"'}, {'OP': '*', 'IS_PUNCT': False}, {'ORTH': '"'}]
    matcher = Matcher(matcher.vocab)
    matcher.add('Quote', [pattern])
    doc = Doc(matcher.vocab, words=words)
    assert len(matcher(doc)) == 1