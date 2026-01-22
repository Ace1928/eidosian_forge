import warnings
import pytest
import srsly
from mock import Mock
from spacy.lang.en import English
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_phrase_matcher_string_attrs_negative(en_vocab):
    """Test that token with the control codes as ORTH are *not* matched."""
    words1 = ['I', 'like', 'cats']
    pos1 = ['PRON', 'VERB', 'NOUN']
    words2 = ['matcher:POS-PRON', 'matcher:POS-VERB', 'matcher:POS-NOUN']
    pos2 = ['X', 'X', 'X']
    pattern = Doc(en_vocab, words=words1, pos=pos1)
    matcher = PhraseMatcher(en_vocab, attr='POS')
    matcher.add('TEST', [pattern])
    doc = Doc(en_vocab, words=words2, pos=pos2)
    matches = matcher(doc)
    assert len(matches) == 0