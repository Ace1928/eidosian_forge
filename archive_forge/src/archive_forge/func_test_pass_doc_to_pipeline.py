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
def test_pass_doc_to_pipeline(nlp, n_process):
    texts = ['cats', 'dogs', 'guinea pigs']
    docs = [nlp.make_doc(text) for text in texts]
    assert not any((len(doc.cats) for doc in docs))
    doc = nlp(docs[0])
    assert doc.text == texts[0]
    assert len(doc.cats) > 0
    if isinstance(get_current_ops(), NumpyOps) or n_process < 2:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            docs = nlp.pipe(docs, n_process=n_process)
            assert [doc.text for doc in docs] == texts
            assert all((len(doc.cats) for doc in docs))