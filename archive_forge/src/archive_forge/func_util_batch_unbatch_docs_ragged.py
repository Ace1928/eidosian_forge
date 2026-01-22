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
def util_batch_unbatch_docs_ragged(model: Model[List[Doc], Ragged], in_data: List[Doc], out_data: Ragged):
    with data_validation(True):
        model.initialize(in_data, out_data)
        Y_batched = model.predict(in_data).data.tolist()
        Y_not_batched = []
        for u in in_data:
            Y_not_batched.extend(model.predict([u]).data.tolist())
        assert_almost_equal(Y_batched, Y_not_batched, decimal=4)