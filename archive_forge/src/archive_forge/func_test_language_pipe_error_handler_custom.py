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
def test_language_pipe_error_handler_custom(en_vocab, n_process):
    """Test the error handling of a custom component that has no pipe method"""
    Language.component('my_evil_component', func=evil_component)
    ops = get_current_ops()
    if isinstance(ops, NumpyOps) or n_process < 2:
        nlp = English()
        nlp.add_pipe('my_evil_component')
        texts = ['TEXT 111', 'TEXT 222', 'TEXT 333', 'TEXT 342', 'TEXT 666']
        with pytest.raises(ValueError):
            list(nlp.pipe(texts))
        nlp.set_error_handler(warn_error)
        logger = logging.getLogger('spacy')
        with mock.patch.object(logger, 'warning') as mock_warning:
            docs = list(nlp.pipe(texts, n_process=n_process))
            if n_process == 1:
                mock_warning.assert_called()
                assert mock_warning.call_count == 2
                assert len(docs) + mock_warning.call_count == len(texts)
            assert [doc.text for doc in docs] == ['TEXT 111', 'TEXT 333', 'TEXT 666']