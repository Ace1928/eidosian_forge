import numpy
import pytest
from numpy.testing import assert_array_equal
from thinc.api import get_current_ops
from spacy.attrs import LENGTH, ORTH
from spacy.lang.en import English
from spacy.tokens import Doc, Span, Token
from spacy.util import filter_spans
from spacy.vocab import Vocab
from ..util import add_vecs_to_vocab
from .test_underscore import clean_underscore  # noqa: F401
def test_span_api_richcmp_other(en_tokenizer):
    doc1 = en_tokenizer('a b')
    doc2 = en_tokenizer('b c')
    assert not doc1[1:2] == doc1[1]
    assert not doc1[1:2] == doc2[0]
    assert not doc1[1:2] == doc2[0:1]
    assert not doc1[0:1] == doc2