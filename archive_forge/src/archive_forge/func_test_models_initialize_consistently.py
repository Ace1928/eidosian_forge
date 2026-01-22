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
@pytest.mark.parametrize('seed,model_func,kwargs', [(0, build_Tok2Vec_model, get_tok2vec_kwargs()), (0, build_bow_text_classifier, get_textcat_bow_kwargs()), (0, build_simple_cnn_text_classifier, get_textcat_cnn_kwargs())])
def test_models_initialize_consistently(seed, model_func, kwargs):
    fix_random_seed(seed)
    model1 = model_func(**kwargs)
    model1.initialize()
    fix_random_seed(seed)
    model2 = model_func(**kwargs)
    model2.initialize()
    params1 = get_all_params(model1)
    params2 = get_all_params(model2)
    assert_array_equal(model1.ops.to_numpy(params1), model2.ops.to_numpy(params2))