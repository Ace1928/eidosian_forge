import logging
import numpy as np
import os
import sys
from typing import Any, Optional
from ray.rllib.utils.annotations import DeveloperAPI, PublicAPI
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.typing import TensorShape, TensorType
@PublicAPI
def try_import_torch(error: bool=False):
    """Tries importing torch and returns the module (or None).

    Args:
        error: Whether to raise an error if torch cannot be imported.

    Returns:
        Tuple consisting of the torch- AND torch.nn modules.

    Raises:
        ImportError: If error=True and PyTorch is not installed.
    """
    if 'RLLIB_TEST_NO_TORCH_IMPORT' in os.environ:
        logger.warning('Not importing PyTorch for test purposes.')
        return _torch_stubs()
    try:
        import torch
        import torch.nn as nn
        return (torch, nn)
    except ImportError:
        if error:
            raise ImportError('Could not import PyTorch! RLlib requires you to install at least one deep-learning framework: `pip install [torch|tensorflow|jax]`.')
        return _torch_stubs()