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
def test_dot_in_factory_names(nlp):
    Language.component('my_evil_component', func=evil_component)
    nlp.add_pipe('my_evil_component')
    with pytest.raises(ValueError, match='not permitted'):
        Language.component('my.evil.component.v1', func=evil_component)
    with pytest.raises(ValueError, match='not permitted'):
        Language.factory('my.evil.component.v1', func=evil_component)