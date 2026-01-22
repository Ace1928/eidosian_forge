import abc
import sys
import traceback
import warnings
from io import StringIO
from decorator import decorator
from traitlets.config.configurable import Configurable
from .getipython import get_ipython
from ..utils.sentinel import Sentinel
from ..utils.dir2 import get_real_method
from ..lib import pretty
from traitlets import (
from typing import Any
def format_display_data(obj, include=None, exclude=None):
    """Return a format data dict for an object.

    By default all format types will be computed.

    Parameters
    ----------
    obj : object
        The Python object whose format data will be computed.

    Returns
    -------
    format_dict : dict
        A dictionary of key/value pairs, one or each format that was
        generated for the object. The keys are the format types, which
        will usually be MIME type strings and the values and JSON'able
        data structure containing the raw data for the representation in
        that format.
    include : list or tuple, optional
        A list of format type strings (MIME types) to include in the
        format data dict. If this is set *only* the format types included
        in this list will be computed.
    exclude : list or tuple, optional
        A list of format type string (MIME types) to exclude in the format
        data dict. If this is set all format types will be computed,
        except for those included in this argument.
    """
    from .interactiveshell import InteractiveShell
    return InteractiveShell.instance().display_formatter.format(obj, include, exclude)