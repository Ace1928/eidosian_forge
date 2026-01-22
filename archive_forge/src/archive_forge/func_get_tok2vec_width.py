from typing import List, Optional, Union, cast
from thinc.api import (
from thinc.types import Floats2d, Ints1d, Ints2d, Ragged
from ...attrs import intify_attr
from ...errors import Errors
from ...ml import _character_embed
from ...pipeline.tok2vec import Tok2VecListener
from ...tokens import Doc
from ...util import registry
from ..featureextractor import FeatureExtractor
from ..staticvectors import StaticVectors
def get_tok2vec_width(model: Model):
    nO = None
    if model.has_ref('tok2vec'):
        tok2vec = model.get_ref('tok2vec')
        if tok2vec.has_dim('nO'):
            nO = tok2vec.get_dim('nO')
        elif tok2vec.has_ref('listener'):
            nO = tok2vec.get_ref('listener').get_dim('nO')
    return nO