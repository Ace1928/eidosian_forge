import inspect
import warnings
import re
import os
import collections
from itertools import islice
from tokenize import open as open_py_source
from .logger import pformat
def format_call(func, args, kwargs, object_name='Memory'):
    """ Returns a nicely formatted statement displaying the function
        call with the given arguments.
    """
    path, signature = format_signature(func, *args, **kwargs)
    msg = '%s\n[%s] Calling %s...\n%s' % (80 * '_', object_name, path, signature)
    return msg