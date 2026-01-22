import warnings
import pytest
import srsly
from mock import Mock
from spacy.lang.en import English
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_phrase_matcher_deprecated(en_vocab):
    matcher = PhraseMatcher(en_vocab)
    matcher.add('TEST', [Doc(en_vocab, words=['helllo'])])
    doc = Doc(en_vocab, words=['hello', 'world'])
    with pytest.warns(DeprecationWarning) as record:
        for _ in matcher.pipe([doc]):
            pass
        assert record.list
        assert 'spaCy v3.0' in str(record.list[0].message)