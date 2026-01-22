import warnings
import weakref
import numpy
import pytest
from numpy.testing import assert_array_equal
from thinc.api import NumpyOps, get_current_ops
from spacy.attrs import (
from spacy.lang.en import English
from spacy.lang.xx import MultiLanguage
from spacy.language import Language
from spacy.lexeme import Lexeme
from spacy.tokens import Doc, Span, SpanGroup, Token
from spacy.vocab import Vocab
from .test_underscore import clean_underscore  # noqa: F401
@pytest.mark.issue(11499)
def test_init_args_unmodified(en_vocab):
    words = ['A', 'sentence']
    ents = ['B-TYPE1', '']
    sent_starts = [True, False]
    Doc(vocab=en_vocab, words=words, ents=ents, sent_starts=sent_starts)
    assert ents == ['B-TYPE1', '']
    assert sent_starts == [True, False]