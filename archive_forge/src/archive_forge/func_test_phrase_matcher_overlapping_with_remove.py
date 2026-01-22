import warnings
import pytest
import srsly
from mock import Mock
from spacy.lang.en import English
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_phrase_matcher_overlapping_with_remove(en_vocab):
    matcher = PhraseMatcher(en_vocab)
    matcher.add('TEST', [Doc(en_vocab, words=['like'])])
    matcher.add('TEST2', [Doc(en_vocab, words=['like'])])
    doc = Doc(en_vocab, words=['I', 'like', 'Google', 'Now', 'best'])
    assert 'TEST' in matcher
    assert len(matcher) == 2
    assert len(matcher(doc)) == 2
    matcher.remove('TEST')
    assert 'TEST' not in matcher
    assert len(matcher) == 1
    assert len(matcher(doc)) == 1
    assert matcher(doc)[0][0] == en_vocab.strings['TEST2']
    matcher.remove('TEST2')
    assert 'TEST2' not in matcher
    assert len(matcher) == 0
    assert len(matcher(doc)) == 0