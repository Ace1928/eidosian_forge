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
def get_tok2vec_kwargs():
    return {'embed': MultiHashEmbed(width=32, rows=[500, 500, 500], attrs=['NORM', 'PREFIX', 'SHAPE'], include_static_vectors=False), 'encode': MaxoutWindowEncoder(width=32, depth=2, maxout_pieces=2, window_size=1)}