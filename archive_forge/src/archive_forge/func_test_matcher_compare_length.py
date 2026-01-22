import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
@pytest.mark.parametrize('cmp, bad', [('==', ['a', 'aaa']), ('!=', ['aa']), ('>=', ['a']), ('<=', ['aaa']), ('>', ['a', 'aa']), ('<', ['aa', 'aaa'])])
def test_matcher_compare_length(en_vocab, cmp, bad):
    matcher = Matcher(en_vocab)
    pattern = [{'LENGTH': {cmp: 2}}]
    matcher.add('LENGTH_COMPARE', [pattern])
    doc = Doc(en_vocab, words=['a', 'aa', 'aaa'])
    matches = matcher(doc)
    assert len(matches) == len(doc) - len(bad)
    doc = Doc(en_vocab, words=bad)
    matches = matcher(doc)
    assert len(matches) == 0