import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_matcher_basic_check(en_vocab):
    matcher = Matcher(en_vocab)
    pattern = [{'TEXT': 'hello'}, {'TEXT': 'world'}]
    with pytest.raises(ValueError):
        matcher.add('TEST', pattern)