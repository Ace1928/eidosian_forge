from typing import TYPE_CHECKING, List, Optional, Union
import torch
from outlines.integrations.llamacpp import (  # noqa: F401
def llamacpp(model_path: str, device: Optional[str]=None, **model_kwargs) -> LlamaCpp:
    from llama_cpp import Llama
    if device == 'cuda':
        model_kwargs['n_gpu_layers'].setdefault(-1)
    model = Llama(model_path, **model_kwargs)
    return LlamaCpp(model=model)