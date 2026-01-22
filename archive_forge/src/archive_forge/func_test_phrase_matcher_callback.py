import warnings
import pytest
import srsly
from mock import Mock
from spacy.lang.en import English
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_phrase_matcher_callback(en_vocab):
    mock = Mock()
    doc = Doc(en_vocab, words=['I', 'like', 'Google', 'Now', 'best'])
    pattern = Doc(en_vocab, words=['Google', 'Now'])
    matcher = PhraseMatcher(en_vocab)
    matcher.add('COMPANY', [pattern], on_match=mock)
    matches = matcher(doc)
    mock.assert_called_once_with(matcher, doc, 0, matches)