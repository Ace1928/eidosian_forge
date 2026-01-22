import warnings
import pytest
import srsly
from mock import Mock
from spacy.lang.en import English
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_phrase_matcher_as_spans(en_vocab):
    """Test the new as_spans=True API."""
    matcher = PhraseMatcher(en_vocab)
    matcher.add('A', [Doc(en_vocab, words=['hello', 'world'])])
    matcher.add('B', [Doc(en_vocab, words=['test'])])
    doc = Doc(en_vocab, words=['...', 'hello', 'world', 'this', 'is', 'a', 'test'])
    matches = matcher(doc, as_spans=True)
    assert len(matches) == 2
    assert isinstance(matches[0], Span)
    assert matches[0].text == 'hello world'
    assert matches[0].label_ == 'A'
    assert isinstance(matches[1], Span)
    assert matches[1].text == 'test'
    assert matches[1].label_ == 'B'