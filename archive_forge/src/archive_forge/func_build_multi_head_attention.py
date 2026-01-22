from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Union
from xformers.utils import import_all_modules
from .activations import Activation, build_activation  # noqa
from .attention import Attention, build_attention  # noqa
from .input_projection import InputProjection, InputProjectionConfig  # noqa
from .multi_head_dispatch import MultiHeadDispatch  # noqa
from .multi_head_dispatch import MultiHeadDispatchConfig
from .patch_embedding import PatchEmbeddingConfig  # noqa
from .patch_embedding import build_patch_embedding  # noqa
from .residual import NormalizationType  # noqa
from .residual import PostNorm  # noqa
from .residual import PreNorm  # noqa
from .residual import RequiresWrappedInputs  # noqa
from .residual import Residual  # noqa
from .residual import ResidualNormStyle  # noqa
import_all_modules(str(Path(__file__).parent), "xformers.components")
def build_multi_head_attention(multi_head_config: Union[MultiHeadDispatchConfig, Dict[str, Any]]):
    """Builds a multihead attention from a config.

    This assumes a 'name' key in the config which is used to determine what
    attention class to instantiate. For instance, a config `{"name": "my_attention",
    "foo": "bar"}` will find a class that was registered as "my_attention"
    (see :func:`register_attention`) and call .from_config on it."""
    if not isinstance(multi_head_config, MultiHeadDispatchConfig):
        field_names = list(map(lambda x: x.name, fields(MultiHeadDispatchConfig)))
        for k in field_names:
            if k not in multi_head_config.keys():
                multi_head_config[k] = None
        if not isinstance(multi_head_config['attention'], Attention):
            if 'num_heads' not in multi_head_config['attention']:
                multi_head_config['attention']['num_heads'] = multi_head_config['num_heads']
            if 'dim_model' not in multi_head_config['attention']:
                multi_head_config['attention']['dim_model'] = multi_head_config['dim_model']
            if 'dim_features' not in multi_head_config['attention'] or multi_head_config['attention']['dim_features'] is None:
                multi_head_config['attention']['dim_features'] = multi_head_config['dim_model'] // multi_head_config['num_heads']
            multi_head_config['attention'] = build_attention(multi_head_config['attention'])
        multi_head_config = MultiHeadDispatchConfig(**multi_head_config)
    return MultiHeadDispatch.from_config(multi_head_config)