import re
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Union
import srsly
from thinc.api import (
from thinc.config import ConfigValidationError
from wasabi import Printer
from ..errors import Errors
from ..schemas import ConfigSchemaPretrain
from ..tokens import Doc
from ..util import dot_to_object, load_model_from_config, registry
from .example import Example
def create_pretraining_model(nlp, pretrain_config):
    """Define a network for the pretraining. We simply add an output layer onto
    the tok2vec input model. The tok2vec input model needs to be a model that
    takes a batch of Doc objects (as a list), and returns a list of arrays.
    Each array in the output needs to have one row per token in the doc.
    The actual tok2vec layer is stored as a reference, and only this bit will be
    serialized to file and read back in when calling the 'train' command.
    """
    with nlp.select_pipes(enable=[]):
        nlp.initialize()
    tok2vec = get_tok2vec_ref(nlp, pretrain_config)
    if type(tok2vec).__name__ == 'Tok2VecListener':
        original_tok2vec = tok2vec.upstream_name if tok2vec.upstream_name != '*' else 'tok2vec'
        tok2vec = nlp.get_pipe(original_tok2vec).model
    try:
        tok2vec.initialize(X=[nlp.make_doc('Give it a doc to infer shapes')])
    except ValueError:
        component = pretrain_config['component']
        layer = pretrain_config['layer']
        raise ValueError(Errors.E874.format(component=component, layer=layer))
    create_function = pretrain_config['objective']
    model = create_function(nlp.vocab, tok2vec)
    model.initialize(X=[nlp.make_doc('Give it a doc to infer shapes')])
    set_dropout_rate(model, pretrain_config['dropout'])
    return model