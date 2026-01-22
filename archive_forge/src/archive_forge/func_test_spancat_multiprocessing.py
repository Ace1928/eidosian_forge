import numpy
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
from thinc.api import NumpyOps, Ragged, get_current_ops
from spacy import util
from spacy.lang.en import English
from spacy.language import Language
from spacy.tokens import SpanGroup
from spacy.tokens._dict_proxies import SpanGroups
from spacy.training import Example
from spacy.util import fix_random_seed, make_tempdir, registry
@pytest.mark.parametrize('name', SPANCAT_COMPONENTS)
@pytest.mark.parametrize('n_process', [1, 2])
def test_spancat_multiprocessing(name, n_process):
    if isinstance(get_current_ops, NumpyOps) or n_process < 2:
        nlp = Language()
        spancat = nlp.add_pipe(name, config={'spans_key': SPAN_KEY})
        train_examples = make_examples(nlp)
        nlp.initialize(get_examples=lambda: train_examples)
        texts = ['Just a sentence.', 'I like London and Berlin', 'I like Berlin', 'I eat ham.']
        docs = list(nlp.pipe(texts, n_process=n_process))
        assert len(docs) == len(texts)