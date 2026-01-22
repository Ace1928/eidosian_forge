import torch.nn as nn
from .imports import is_fp8_available
def has_transformer_engine_layers(model):
    """
    Returns whether a given model has some `transformer_engine` layer or not.
    """
    if not is_fp8_available():
        raise ImportError('Using `has_transformer_engine_layers` requires transformer_engine to be installed.')
    for m in model.modules():
        if isinstance(m, (te.LayerNorm, te.Linear, te.TransformerLayer)):
            return True
    return False