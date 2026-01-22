import inspect
import os.path
import sys
from _pydev_bundle._pydev_tipper_common import do_find
from _pydevd_bundle.pydevd_utils import hasattr_checked, dir_checked
from inspect import getfullargspec
def search_definition(data):
    """@return file, line, col
    """
    data = data.replace('\n', '')
    if data.endswith('.'):
        data = data.rstrip('.')
    f, mod, parent, foundAs = Find(data)
    try:
        return (do_find(f, mod), foundAs)
    except:
        return (do_find(f, parent), foundAs)