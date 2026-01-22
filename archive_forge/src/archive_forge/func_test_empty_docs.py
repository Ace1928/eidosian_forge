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
@pytest.mark.parametrize('model_func,kwargs', [(StaticVectors, {'nO': 128, 'nM': 300})])
def test_empty_docs(model_func, kwargs):
    nlp = English()
    model = model_func(**kwargs).initialize()
    for n_docs in range(3):
        docs = [nlp('') for _ in range(n_docs)]
        model.predict(docs)
        output, backprop = model.begin_update(docs)
        backprop(output)