import warnings
import pytest
import srsly
from mock import Mock
from spacy.lang.en import English
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_span_in_phrasematcher(en_vocab):
    """Ensure that PhraseMatcher accepts Span and Doc as input"""
    words = ['I', 'like', 'Spans', 'and', 'Docs', 'in', 'my', 'input', ',', 'and', 'nothing', 'else', '.']
    doc = Doc(en_vocab, words=words)
    span = doc[:8]
    pattern = Doc(en_vocab, words=['Spans', 'and', 'Docs'])
    matcher = PhraseMatcher(en_vocab)
    matcher.add('SPACY', [pattern])
    matches_doc = matcher(doc)
    matches_span = matcher(span)
    assert len(matches_doc) == 1
    assert len(matches_span) == 1