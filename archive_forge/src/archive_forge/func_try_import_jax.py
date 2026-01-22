import logging
import numpy as np
import os
import sys
from typing import Any, Optional
from ray.rllib.utils.annotations import DeveloperAPI, PublicAPI
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.typing import TensorShape, TensorType
@PublicAPI
def try_import_jax(error: bool=False):
    """Tries importing JAX and FLAX and returns both modules (or Nones).

    Args:
        error: Whether to raise an error if JAX/FLAX cannot be imported.

    Returns:
        Tuple containing the jax- and the flax modules.

    Raises:
        ImportError: If error=True and JAX is not installed.
    """
    if 'RLLIB_TEST_NO_JAX_IMPORT' in os.environ:
        logger.warning('Not importing JAX for test purposes.')
        return (None, None)
    try:
        import jax
        import flax
    except ImportError:
        if error:
            raise ImportError('Could not import JAX! RLlib requires you to install at least one deep-learning framework: `pip install [torch|tensorflow|jax]`.')
        return (None, None)
    return (jax, flax)