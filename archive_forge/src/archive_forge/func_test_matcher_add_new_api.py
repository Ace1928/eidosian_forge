import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_matcher_add_new_api(en_vocab):
    doc = Doc(en_vocab, words=['a', 'b'])
    patterns = [[{'TEXT': 'a'}], [{'TEXT': 'a'}, {'TEXT': 'b'}]]
    matcher = Matcher(en_vocab)
    on_match = Mock()
    matcher = Matcher(en_vocab)
    matcher.add('NEW_API', patterns)
    assert len(matcher(doc)) == 2
    matcher = Matcher(en_vocab)
    on_match = Mock()
    matcher.add('NEW_API_CALLBACK', patterns, on_match=on_match)
    assert len(matcher(doc)) == 2
    assert on_match.call_count == 2