import re
import inspect
import os
import sys
from importlib.machinery import SourceFileLoader
def conf_from_dict(conf_dict):
    """
    Creates a configuration dictionary from a dictionary.

    :param conf_dict: The configuration dictionary.
    """
    conf = Config(filename=conf_dict.get('__file__', ''))
    for k, v in iter(conf_dict.items()):
        if k.startswith('__'):
            continue
        elif inspect.ismodule(v):
            continue
        conf[k] = v
    return conf