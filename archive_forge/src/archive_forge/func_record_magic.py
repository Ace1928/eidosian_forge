import os
import re
import sys
from getopt import getopt, GetoptError
from traitlets.config.configurable import Configurable
from . import oinspect
from .error import UsageError
from .inputtransformer2 import ESC_MAGIC, ESC_MAGIC2
from ..utils.ipstruct import Struct
from ..utils.process import arg_split
from ..utils.text import dedent
from traitlets import Bool, Dict, Instance, observe
from logging import error
import typing as t
def record_magic(dct, magic_kind, magic_name, func):
    """Utility function to store a function as a magic of a specific kind.

    Parameters
    ----------
    dct : dict
        A dictionary with 'line' and 'cell' subdicts.
    magic_kind : str
        Kind of magic to be stored.
    magic_name : str
        Key to store the magic as.
    func : function
        Callable object to store.
    """
    if magic_kind == 'line_cell':
        dct['line'][magic_name] = dct['cell'][magic_name] = func
    else:
        dct[magic_kind][magic_name] = func