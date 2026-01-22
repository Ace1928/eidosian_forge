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
def test_span_groups(en_tokenizer):
    doc = en_tokenizer('Some text about Colombia and the Czech Republic')
    doc.spans['hi'] = [Span(doc, 3, 4, label='bye')]
    assert 'hi' in doc.spans
    assert 'bye' not in doc.spans
    assert len(doc.spans['hi']) == 1
    assert doc.spans['hi'][0].label_ == 'bye'
    doc.spans['hi'].append(doc[0:3])
    assert len(doc.spans['hi']) == 2
    assert doc.spans['hi'][1].text == 'Some text about'
    assert [span.text for span in doc.spans['hi']] == ['Colombia', 'Some text about']
    assert not doc.spans['hi'].has_overlap
    doc.ents = [Span(doc, 3, 4, label='GPE'), Span(doc, 6, 8, label='GPE')]
    doc.spans['hi'].extend(doc.ents)
    assert len(doc.spans['hi']) == 4
    assert [span.label_ for span in doc.spans['hi']] == ['bye', '', 'GPE', 'GPE']
    assert doc.spans['hi'].has_overlap
    del doc.spans['hi']
    assert 'hi' not in doc.spans