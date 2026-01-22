import warnings
import pytest
import srsly
from mock import Mock
from spacy.lang.en import English
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_phrase_matcher_remove(en_vocab):
    matcher = PhraseMatcher(en_vocab)
    matcher.add('TEST1', [Doc(en_vocab, words=['like'])])
    matcher.add('TEST2', [Doc(en_vocab, words=['best'])])
    doc = Doc(en_vocab, words=['I', 'like', 'Google', 'Now', 'best'])
    assert 'TEST1' in matcher
    assert 'TEST2' in matcher
    assert 'TEST3' not in matcher
    assert len(matcher(doc)) == 2
    matcher.remove('TEST1')
    assert 'TEST1' not in matcher
    assert 'TEST2' in matcher
    assert 'TEST3' not in matcher
    assert len(matcher(doc)) == 1
    matcher.remove('TEST2')
    assert 'TEST1' not in matcher
    assert 'TEST2' not in matcher
    assert 'TEST3' not in matcher
    assert len(matcher(doc)) == 0
    with pytest.raises(KeyError):
        matcher.remove('TEST3')
    assert 'TEST1' not in matcher
    assert 'TEST2' not in matcher
    assert 'TEST3' not in matcher
    assert len(matcher(doc)) == 0