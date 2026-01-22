from __future__ import annotations
from .core import Encoding
from .registry import get_encoding
def encoding_name_for_model(model_name: str) -> str:
    """Returns the name of the encoding used by a model.

    Raises a KeyError if the model name is not recognised.
    """
    encoding_name = None
    if model_name in MODEL_TO_ENCODING:
        encoding_name = MODEL_TO_ENCODING[model_name]
    else:
        for model_prefix, model_encoding_name in MODEL_PREFIX_TO_ENCODING.items():
            if model_name.startswith(model_prefix):
                return model_encoding_name
    if encoding_name is None:
        raise KeyError(f'Could not automatically map {model_name} to a tokeniser. Please use `tiktoken.get_encoding` to explicitly get the tokeniser you expect.') from None
    return encoding_name