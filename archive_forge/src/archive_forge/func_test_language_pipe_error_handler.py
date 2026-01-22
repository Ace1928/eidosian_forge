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
@pytest.mark.parametrize('n_process', [1, 2])
def test_language_pipe_error_handler(n_process):
    """Test that the error handling of nlp.pipe works well"""
    ops = get_current_ops()
    if isinstance(ops, NumpyOps) or n_process < 2:
        nlp = English()
        nlp.add_pipe('merge_subtokens')
        nlp.initialize()
        texts = ['Curious to see what will happen to this text.', 'And this one.']
        with pytest.raises(ValueError):
            nlp(texts[0])
        with pytest.raises(ValueError):
            list(nlp.pipe(texts, n_process=n_process))
        nlp.set_error_handler(raise_error)
        with pytest.raises(ValueError):
            list(nlp.pipe(texts, n_process=n_process))
        nlp.set_error_handler(ignore_error)
        docs = list(nlp.pipe(texts, n_process=n_process))
        assert len(docs) == 0
        nlp(texts[0])