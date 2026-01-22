import enum
import os
import sys
from os import getcwd
from os.path import dirname, exists, join
from weakref import ref
from .etsconfig.api import ETSConfig
def traits_home():
    """ Gets the path to the Traits home directory.
    """
    global _traits_home
    if _traits_home is None:
        _traits_home = verify_path(join(ETSConfig.application_data, 'traits'))
    return _traits_home