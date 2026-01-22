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
@pytest.mark.issue(3962)
def test_issue3962(en_vocab):
    """Ensure that as_doc does not result in out-of-bound access of tokens.
    This is achieved by setting the head to itself if it would lie out of the span otherwise."""
    words = ['He', 'jests', 'at', 'scars', ',', 'that', 'never', 'felt', 'a', 'wound', '.']
    heads = [1, 7, 1, 2, 7, 7, 7, 7, 9, 7, 7]
    deps = ['nsubj', 'ccomp', 'prep', 'pobj', 'punct', 'nsubj', 'neg', 'ROOT', 'det', 'dobj', 'punct']
    doc = Doc(en_vocab, words=words, heads=heads, deps=deps)
    span2 = doc[1:5]
    doc2 = span2.as_doc()
    doc2_json = doc2.to_json()
    assert doc2_json
    assert doc2[0].head.text == 'jests'
    assert doc2[0].dep_ == 'dep'
    assert doc2[1].head.text == 'jests'
    assert doc2[1].dep_ == 'prep'
    assert doc2[2].head.text == 'at'
    assert doc2[2].dep_ == 'pobj'
    assert doc2[3].head.text == 'jests'
    assert doc2[3].dep_ == 'dep'
    assert len(list(doc2.sents)) == 1
    span3 = doc[6:9]
    doc3 = span3.as_doc()
    doc3_json = doc3.to_json()
    assert doc3_json
    assert doc3[0].head.text == 'felt'
    assert doc3[0].dep_ == 'neg'
    assert doc3[1].head.text == 'felt'
    assert doc3[1].dep_ == 'ROOT'
    assert doc3[2].head.text == 'felt'
    assert doc3[2].dep_ == 'dep'
    assert len(list(doc3.sents)) == 1