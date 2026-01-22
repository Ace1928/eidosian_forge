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
@registry.architectures('spacy.TextCatParametricAttention.v1')
def build_textcat_parametric_attention_v1(tok2vec: Model[List[Doc], List[Floats2d]], exclusive_classes: bool, nO: Optional[int]=None) -> Model[List[Doc], Floats2d]:
    width = tok2vec.maybe_get_dim('nO')
    parametric_attention = _build_parametric_attention_with_residual_nonlinear(tok2vec=tok2vec, nonlinear_layer=Maxout(nI=width, nO=width), key_transform=Gelu(nI=width, nO=width))
    with Model.define_operators({'>>': chain}):
        if exclusive_classes:
            output_layer = Softmax(nO=nO)
        else:
            output_layer = Linear(nO=nO) >> Logistic()
        model = parametric_attention >> output_layer
    if model.has_dim('nO') is not False and nO is not None:
        model.set_dim('nO', cast(int, nO))
    model.set_ref('output_layer', output_layer)
    model.attrs['multi_label'] = not exclusive_classes
    return model