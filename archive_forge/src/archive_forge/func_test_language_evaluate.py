import itertools
import logging
import warnings
from unittest import mock
import pytest
from thinc.api import CupyOps, NumpyOps, get_current_ops
import spacy
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.language import Language
from spacy.scorer import Scorer
from spacy.tokens import Doc, Span
from spacy.training import Example
from spacy.util import find_matching_language, ignore_error, raise_error, registry
from spacy.vocab import Vocab
from .util import add_vecs_to_vocab, assert_docs_equal
def test_language_evaluate(nlp):
    text = 'hello world'
    annots = {'doc_annotation': {'cats': {'POSITIVE': 1.0, 'NEGATIVE': 0.0}}}
    doc = Doc(nlp.vocab, words=text.split(' '))
    example = Example.from_dict(doc, annots)
    scores = nlp.evaluate([example])
    assert scores['speed'] > 0
    scores = nlp.evaluate((eg for eg in [example]))
    assert scores['speed'] > 0
    with pytest.raises(TypeError):
        nlp.evaluate(example)
    with pytest.raises(TypeError):
        nlp.evaluate([(text, annots)])
    with pytest.raises(TypeError):
        nlp.evaluate([(doc, annots)])
    with pytest.raises(TypeError):
        nlp.evaluate([text, annots])