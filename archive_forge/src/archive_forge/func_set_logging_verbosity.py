import sys
from absl import logging
from keras.src.api_export import keras_export
from keras.src.backend.common import global_state
def set_logging_verbosity(level):
    """Sets the verbosity level for logging.

    Supported log levels are as follows:

    - `"FATAL"` (least verbose)
    - `"ERROR"`
    - `"WARNING"`
    - `"INFO"`
    - `"DEBUG"` (most verbose)

    Args:
        level: A string corresponding to the level of verbosity for logging.
    """
    valid_levels = {'FATAL': logging.FATAL, 'ERROR': logging.ERROR, 'WARNING': logging.WARNING, 'INFO': logging.INFO, 'DEBUG': logging.DEBUG}
    verbosity = valid_levels.get(level)
    if verbosity is None:
        raise ValueError(f'Please pass a valid level for logging verbosity. Expected one of: {set(valid_levels.keys())}. Received: {level}')
    logging.set_verbosity(verbosity)