import logging
import numpy as np
import os
import sys
from typing import Any, Optional
from ray.rllib.utils.annotations import DeveloperAPI, PublicAPI
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.typing import TensorShape, TensorType
@PublicAPI
def try_import_tfp(error: bool=False):
    """Tries importing tfp and returns the module (or None).

    Args:
        error: Whether to raise an error if tfp cannot be imported.

    Returns:
        The tfp module.

    Raises:
        ImportError: If error=True and tfp is not installed.
    """
    if 'RLLIB_TEST_NO_TF_IMPORT' in os.environ:
        logger.warning('Not importing TensorFlow Probability for test purposes.')
        return None
    try:
        import tensorflow_probability as tfp
        return tfp
    except ImportError as e:
        if error:
            raise e
        return None