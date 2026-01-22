from typing import List
import numpy
import pytest
from numpy.testing import assert_almost_equal
from thinc.api import Model, data_validation, get_current_ops
from thinc.types import Array2d, Ragged
from spacy.lang.en import English
from spacy.ml import FeatureExtractor, StaticVectors
from spacy.ml._character_embed import CharacterEmbed
from spacy.tokens import Doc
from spacy.vocab import Vocab
@pytest.mark.parametrize('model,in_data,out_data', LAYERS)
def test_layers_batching_all(model, in_data, out_data):
    if isinstance(in_data, list) and isinstance(in_data[0], Doc):
        if isinstance(out_data, OPS.xp.ndarray) and out_data.ndim == 2:
            util_batch_unbatch_docs_array(model, in_data, out_data)
        elif isinstance(out_data, list) and isinstance(out_data[0], OPS.xp.ndarray) and (out_data[0].ndim == 2):
            util_batch_unbatch_docs_list(model, in_data, out_data)
        elif isinstance(out_data, Ragged):
            util_batch_unbatch_docs_ragged(model, in_data, out_data)