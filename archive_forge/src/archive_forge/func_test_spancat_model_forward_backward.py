from typing import List
import numpy
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from thinc.api import (
from spacy.lang.en import English
from spacy.lang.en.examples import sentences as EN_SENTENCES
from spacy.ml.extract_spans import _get_span_indices, extract_spans
from spacy.ml.models import (
from spacy.ml.staticvectors import StaticVectors
from spacy.util import registry
def test_spancat_model_forward_backward(nO=5):
    tok2vec = build_Tok2Vec_model(**get_tok2vec_kwargs())
    docs = get_docs()
    spans_list = []
    lengths = []
    for doc in docs:
        spans_list.append(doc[:2])
        spans_list.append(doc[1:4])
        lengths.append(2)
    spans = Ragged(tok2vec.ops.asarray([[s.start, s.end] for s in spans_list], dtype='i'), tok2vec.ops.asarray(lengths, dtype='i'))
    model = build_spancat_model(tok2vec, reduce_mean(), chain(Relu(nO=nO), Logistic())).initialize(X=(docs, spans))
    Y, backprop = model((docs, spans), is_train=True)
    assert Y.shape == (spans.dataXd.shape[0], nO)
    backprop(Y)