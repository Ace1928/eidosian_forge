import copy
import pickle
import sys
import typing
import warnings
from types import FunctionType
from traitlets.log import get_logger
from traitlets.utils.importstring import import_item
def can_sequence(obj):
    """can the elements of a sequence"""
    if istype(obj, sequence_types):
        t = type(obj)
        return t([can(i) for i in obj])
    return obj