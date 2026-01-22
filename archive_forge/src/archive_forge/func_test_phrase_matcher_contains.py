import warnings
import pytest
import srsly
from mock import Mock
from spacy.lang.en import English
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_phrase_matcher_contains(en_vocab):
    matcher = PhraseMatcher(en_vocab)
    matcher.add('TEST', [Doc(en_vocab, words=['test'])])
    assert 'TEST' in matcher
    assert 'TEST2' not in matcher