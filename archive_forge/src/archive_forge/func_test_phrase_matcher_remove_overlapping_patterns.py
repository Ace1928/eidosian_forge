import warnings
import pytest
import srsly
from mock import Mock
from spacy.lang.en import English
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_phrase_matcher_remove_overlapping_patterns(en_vocab):
    matcher = PhraseMatcher(en_vocab)
    pattern1 = Doc(en_vocab, words=['this'])
    pattern2 = Doc(en_vocab, words=['this', 'is'])
    pattern3 = Doc(en_vocab, words=['this', 'is', 'a'])
    pattern4 = Doc(en_vocab, words=['this', 'is', 'a', 'word'])
    matcher.add('THIS', [pattern1, pattern2, pattern3, pattern4])
    matcher.remove('THIS')