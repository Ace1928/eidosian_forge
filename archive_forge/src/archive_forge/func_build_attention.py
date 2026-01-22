import logging
from pathlib import Path
from typing import Any, Callable, Dict, Set, Union
import torch
from xformers.utils import (
from ._sputnik_sparse import SparseCS
from .attention_mask import AttentionMask
from .base import Attention, AttentionConfig  # noqa
from .favor import FavorAttention  # noqa
from .global_tokens import GlobalAttention  # noqa
from .linformer import LinformerAttention  # noqa
from .local import LocalAttention  # noqa
from .nystrom import NystromAttention  # noqa
from .ortho import OrthoFormerAttention  # noqa
from .random import RandomAttention  # noqa
from .scaled_dot_product import ScaledDotProduct  # noqa
import_all_modules(str(Path(__file__).parent), "xformers.components.attention")
def build_attention(config: Union[Dict[str, Any], AttentionConfig]):
    """Builds an attention from a config.

    This assumes a 'name' key in the config which is used to determine what
    attention class to instantiate. For instance, a config `{"name": "my_attention",
    "foo": "bar"}` will find a class that was registered as "my_attention"
    (see :func:`register_attention`) and call .from_config on it."""
    if not isinstance(config, AttentionConfig):
        try:
            config_instance = generate_matching_config(config, ATTENTION_REGISTRY[config['name']].config)
        except KeyError as e:
            name = config['name']
            logger.warning(f'{name} not available among {ATTENTION_REGISTRY.keys()}')
            raise e
    else:
        config_instance = config
    return ATTENTION_REGISTRY[config_instance.name].constructor.from_config(config_instance)