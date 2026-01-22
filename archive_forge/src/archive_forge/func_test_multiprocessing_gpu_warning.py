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
@pytest.mark.skipif(not isinstance(get_current_ops(), CupyOps), reason='test requires GPU')
def test_multiprocessing_gpu_warning(nlp2, texts):
    texts = texts * 10
    docs = nlp2.pipe(texts, n_process=2, batch_size=2)
    with pytest.warns(UserWarning, match='multiprocessing with GPU models'):
        with pytest.raises(ValueError):
            for _ in docs:
                pass