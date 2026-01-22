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
def test_doc_api_getitem(en_tokenizer):
    text = 'Give it back! He pleaded.'
    tokens = en_tokenizer(text)
    assert tokens[0].text == 'Give'
    assert tokens[-1].text == '.'
    with pytest.raises(IndexError):
        tokens[len(tokens)]

    def to_str(span):
        return '/'.join((token.text for token in span))
    span = tokens[1:1]
    assert not to_str(span)
    span = tokens[1:4]
    assert to_str(span) == 'it/back/!'
    span = tokens[1:4:1]
    assert to_str(span) == 'it/back/!'
    with pytest.raises(ValueError):
        tokens[1:4:2]
    with pytest.raises(ValueError):
        tokens[1:4:-1]
    span = tokens[-3:6]
    assert to_str(span) == 'He/pleaded'
    span = tokens[4:-1]
    assert to_str(span) == 'He/pleaded'
    span = tokens[-5:-3]
    assert to_str(span) == 'back/!'
    span = tokens[5:4]
    assert span.start == span.end == 5 and (not to_str(span))
    span = tokens[4:-3]
    assert span.start == span.end == 4 and (not to_str(span))
    span = tokens[:]
    assert to_str(span) == 'Give/it/back/!/He/pleaded/.'
    span = tokens[4:]
    assert to_str(span) == 'He/pleaded/.'
    span = tokens[:4]
    assert to_str(span) == 'Give/it/back/!'
    span = tokens[:-3]
    assert to_str(span) == 'Give/it/back/!'
    span = tokens[-3:]
    assert to_str(span) == 'He/pleaded/.'
    span = tokens[4:50]
    assert to_str(span) == 'He/pleaded/.'
    span = tokens[-50:4]
    assert to_str(span) == 'Give/it/back/!'
    span = tokens[-50:-40]
    assert span.start == span.end == 0 and (not to_str(span))
    span = tokens[40:50]
    assert span.start == span.end == 7 and (not to_str(span))
    span = tokens[1:4]
    assert span[0].orth_ == 'it'
    subspan = span[:]
    assert to_str(subspan) == 'it/back/!'
    subspan = span[:2]
    assert to_str(subspan) == 'it/back'
    subspan = span[1:]
    assert to_str(subspan) == 'back/!'
    subspan = span[:-1]
    assert to_str(subspan) == 'it/back'
    subspan = span[-2:]
    assert to_str(subspan) == 'back/!'
    subspan = span[1:2]
    assert to_str(subspan) == 'back'
    subspan = span[-2:-1]
    assert to_str(subspan) == 'back'
    subspan = span[-50:50]
    assert to_str(subspan) == 'it/back/!'
    subspan = span[50:-50]
    assert subspan.start == subspan.end == 4 and (not to_str(subspan))