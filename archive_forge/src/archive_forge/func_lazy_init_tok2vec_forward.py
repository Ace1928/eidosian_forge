from typing import List
from thinc.api import Model
from thinc.types import Floats2d
from spacy.tokens import Doc
from spacy.util import registry
def lazy_init_tok2vec_forward(model: Model, X: List[Doc], is_train: bool):
    width = model.get_dim('nO')
    Y = [model.ops.alloc2f(len(doc), width) for doc in X]

    def backprop(dY):
        return []
    return (Y, backprop)