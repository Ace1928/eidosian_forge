import warnings
import pytest
import srsly
from mock import Mock
from spacy.lang.en import English
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_phrase_matcher_length(en_vocab):
    matcher = PhraseMatcher(en_vocab)
    assert len(matcher) == 0
    matcher.add('TEST', [Doc(en_vocab, words=['test'])])
    assert len(matcher) == 1
    matcher.add('TEST2', [Doc(en_vocab, words=['test2'])])
    assert len(matcher) == 2