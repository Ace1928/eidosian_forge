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
def test_language_pipe_stream(nlp2, n_process, texts):
    ops = get_current_ops()
    if isinstance(ops, NumpyOps) or n_process < 2:
        stream_texts = itertools.cycle(texts)
        texts0, texts1 = itertools.tee(stream_texts)
        expecteds = (nlp2(text) for text in texts0)
        docs = nlp2.pipe(texts1, n_process=n_process, batch_size=2)
        n_fetch = 20
        for doc, expected_doc in itertools.islice(zip(docs, expecteds), n_fetch):
            assert_docs_equal(doc, expected_doc)