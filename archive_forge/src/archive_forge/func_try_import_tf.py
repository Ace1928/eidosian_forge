import logging
import numpy as np
import os
import sys
from typing import Any, Optional
from ray.rllib.utils.annotations import DeveloperAPI, PublicAPI
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.typing import TensorShape, TensorType
@PublicAPI
def try_import_tf(error: bool=False):
    """Tries importing tf and returns the module (or None).

    Args:
        error: Whether to raise an error if tf cannot be imported.

    Returns:
        Tuple containing
        1) tf1.x module (either from tf2.x.compat.v1 OR as tf1.x).
        2) tf module (resulting from `import tensorflow`). Either tf1.x or
        2.x. 3) The actually installed tf version as int: 1 or 2.

    Raises:
        ImportError: If error=True and tf is not installed.
    """
    tf_stub = _TFStub()
    if 'RLLIB_TEST_NO_TF_IMPORT' in os.environ:
        logger.warning('Not importing TensorFlow for test purposes')
        return (None, tf_stub, None)
    if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    was_imported = False
    if 'tensorflow' in sys.modules:
        tf_module = sys.modules['tensorflow']
        was_imported = True
    else:
        try:
            import tensorflow as tf_module
        except ImportError:
            if error:
                raise ImportError('Could not import TensorFlow! RLlib requires you to install at least one deep-learning framework: `pip install [torch|tensorflow|jax]`.')
            return (None, tf_stub, None)
    try:
        tf1_module = tf_module.compat.v1
        tf1_module.logging.set_verbosity(tf1_module.logging.ERROR)
        if not was_imported:
            tf1_module.disable_v2_behavior()
            tf1_module.enable_resource_variables()
        tf1_module.logging.set_verbosity(tf1_module.logging.WARN)
    except AttributeError:
        tf1_module = tf_module
    if not hasattr(tf_module, '__version__'):
        version = 1
    else:
        version = 2 if '2.' in tf_module.__version__[:2] else 1
    return (tf1_module, tf_module, version)