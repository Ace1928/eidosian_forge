import enum
import os
import sys
from os import getcwd
from os.path import dirname, exists, join
from weakref import ref
from .etsconfig.api import ETSConfig
def verify_path(path):
    """ Verify that a specified path exists, and try to create it if it
        does not exist.
    """
    if not exists(path):
        try:
            os.mkdir(path)
        except:
            pass
    return path