from functools import partial
from typing import List, Optional, Tuple, cast
from thinc.api import (
from thinc.layers.chain import init as init_chain
from thinc.layers.resizable import resize_linear_weighted, resize_model
from thinc.types import ArrayXd, Floats2d
from ...attrs import ORTH
from ...errors import Errors
from ...tokens import Doc
from ...util import registry
from ..extract_ngrams import extract_ngrams
from ..staticvectors import StaticVectors
from .tok2vec import get_tok2vec_width
def resize_and_set_ref(model, new_nO, resizable_layer):
    resizable_layer = resize_model(resizable_layer, new_nO)
    model.set_ref('output_layer', resizable_layer.layers[0])
    model.set_dim('nO', new_nO, force=True)
    return model