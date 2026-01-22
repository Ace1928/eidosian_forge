import re
import inspect
import os
import sys
from importlib.machinery import SourceFileLoader
def get_conf_path_from_env():
    """
    If the ``PECAN_CONFIG`` environment variable exists and it points to
    a valid path it will return that, otherwise it will raise
    a ``RuntimeError``.
    """
    config_path = os.environ.get('PECAN_CONFIG')
    if not config_path:
        error = 'PECAN_CONFIG is not set and no config file was passed as an argument.'
    elif not os.path.isfile(config_path):
        error = 'PECAN_CONFIG was set to an invalid path: %s' % config_path
    else:
        return config_path
    raise RuntimeError(error)