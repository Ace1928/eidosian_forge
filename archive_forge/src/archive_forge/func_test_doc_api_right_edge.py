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
def test_doc_api_right_edge(en_vocab):
    """Test for bug occurring from Unshift action, causing incorrect right edge"""
    words = ['I', 'have', 'proposed', 'to', 'myself', ',', 'for', 'the', 'sake', 'of', 'such', 'as', 'live', 'under', 'the', 'government', 'of', 'the', 'Romans', ',', 'to', 'translate', 'those', 'books', 'into', 'the', 'Greek', 'tongue', '.']
    heads = [2, 2, 2, 2, 3, 2, 21, 8, 6, 8, 11, 8, 11, 12, 15, 13, 15, 18, 16, 12, 21, 2, 23, 21, 21, 27, 27, 24, 2]
    deps = ['dep'] * len(heads)
    doc = Doc(en_vocab, words=words, heads=heads, deps=deps)
    assert doc[6].text == 'for'
    subtree = [w.text for w in doc[6].subtree]
    assert subtree == ['for', 'the', 'sake', 'of', 'such', 'as', 'live', 'under', 'the', 'government', 'of', 'the', 'Romans', ',']
    assert doc[6].right_edge.text == ','