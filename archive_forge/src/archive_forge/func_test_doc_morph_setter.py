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
def test_doc_morph_setter(en_tokenizer, de_tokenizer):
    doc1 = en_tokenizer('a b')
    doc1b = en_tokenizer('c d')
    doc2 = de_tokenizer('a b')
    doc1[0].morph = doc1[1].morph
    assert doc1[0].morph.key == 0
    assert doc1[1].morph.key == 0
    doc1[0].set_morph('Feat=Val')
    doc1[1].morph = doc1[0].morph
    assert doc1[0].morph == doc1[1].morph
    doc1b[0].morph = doc1[0].morph
    assert doc1[0].morph == doc1b[0].morph
    doc2[0].set_morph('Feat2=Val2')
    with pytest.raises(ValueError):
        doc1[0].morph = doc2[0].morph