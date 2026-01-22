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
def util_batch_unbatch_docs_list(model: Model[List[Doc], List[Array2d]], in_data: List[Doc], out_data: List[Array2d]):
    with data_validation(True):
        model.initialize(in_data, out_data)
        Y_batched = model.predict(in_data)
        Y_not_batched = [model.predict([u])[0] for u in in_data]
        for i in range(len(Y_batched)):
            assert_almost_equal(OPS.to_numpy(Y_batched[i]), OPS.to_numpy(Y_not_batched[i]), decimal=4)