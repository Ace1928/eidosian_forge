import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_matcher_match_multi(matcher):
    words = ['I', 'like', 'Google', 'Now', 'and', 'java', 'best']
    doc = Doc(matcher.vocab, words=words)
    assert matcher(doc) == [(doc.vocab.strings['GoogleNow'], 2, 4), (doc.vocab.strings['Java'], 5, 6)]