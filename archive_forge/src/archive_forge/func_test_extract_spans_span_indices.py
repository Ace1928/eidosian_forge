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
def test_extract_spans_span_indices():
    model = extract_spans().initialize()
    spans = Ragged(model.ops.asarray([[0, 3], [2, 3], [5, 7]], dtype='i'), model.ops.asarray([2, 1], dtype='i'))
    x_lengths = model.ops.asarray([5, 10], dtype='i')
    indices = _get_span_indices(model.ops, spans, x_lengths)
    assert list(indices) == [0, 1, 2, 2, 10, 11]