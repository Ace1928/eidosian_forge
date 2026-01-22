from typing import Callable, Optional, Union
from ray.rllib.core.models.specs.specs_base import TensorSpec
from ray.rllib.core.models.specs.specs_dict import SpecDict
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.framework import try_import_jax, try_import_tf, try_import_torch
@DeveloperAPI
def get_activation_fn(name: Optional[Union[Callable, str]]=None, framework: str='tf'):
    """Returns a framework specific activation function, given a name string.

    Args:
        name: One of "relu" (default), "tanh", "elu",
            "swish" (or "silu", which is the same), or "linear" (same as None).
        framework: One of "jax", "tf|tf2" or "torch".

    Returns:
        A framework-specific activtion function. e.g. tf.nn.tanh or
            torch.nn.ReLU. None if name in ["linear", None].

    Raises:
        ValueError: If name is an unknown activation function.
    """
    if callable(name):
        return name
    name_lower = name.lower() if isinstance(name, str) else name
    if framework == 'torch':
        if name_lower in ['linear', None]:
            return None
        _, nn = try_import_torch()
        fn = getattr(nn, name, None)
        if fn is not None:
            return fn
        if name_lower in ['swish', 'silu']:
            return nn.SiLU
        elif name_lower == 'relu':
            return nn.ReLU
        elif name_lower == 'tanh':
            return nn.Tanh
        elif name_lower == 'elu':
            return nn.ELU
    elif framework == 'jax':
        if name_lower in ['linear', None]:
            return None
        jax, _ = try_import_jax()
        if name_lower in ['swish', 'silu']:
            return jax.nn.swish
        if name_lower == 'relu':
            return jax.nn.relu
        elif name_lower == 'tanh':
            return jax.nn.hard_tanh
        elif name_lower == 'elu':
            return jax.nn.elu
    else:
        assert framework in ['tf', 'tf2'], 'Unsupported framework `{}`!'.format(framework)
        if name_lower in ['linear', None]:
            return None
        tf1, tf, tfv = try_import_tf()
        fn = getattr(tf.nn, name_lower, None)
        if fn is not None:
            return fn
    raise ValueError('Unknown activation ({}) for framework={}!'.format(name, framework))