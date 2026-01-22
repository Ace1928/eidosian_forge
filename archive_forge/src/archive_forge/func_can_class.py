import copy
import pickle
import sys
import typing
import warnings
from types import FunctionType
from traitlets.log import get_logger
from traitlets.utils.importstring import import_item
def can_class(obj):
    """Can a class object."""
    if isinstance(obj, class_type) and obj.__module__ == '__main__':
        return CannedClass(obj)
    return obj