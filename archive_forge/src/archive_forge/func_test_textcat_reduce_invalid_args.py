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
def test_textcat_reduce_invalid_args():
    textcat_reduce = registry.architectures.get('spacy.TextCatReduce.v1')
    tok2vec = make_test_tok2vec()
    with pytest.raises(ValueError, match='must be used with at least one reduction'):
        textcat_reduce(tok2vec=tok2vec, exclusive_classes=False, use_reduce_first=False, use_reduce_last=False, use_reduce_max=False, use_reduce_mean=False)