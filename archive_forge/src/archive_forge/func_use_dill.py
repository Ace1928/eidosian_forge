import copy
import pickle
import sys
import typing
import warnings
from types import FunctionType
from traitlets.log import get_logger
from traitlets.utils.importstring import import_item
def use_dill():
    """use dill to expand serialization support

    adds support for object methods and closures to serialization.
    """
    import dill
    global pickle
    pickle = dill
    try:
        from ipykernel import serialize
    except ImportError:
        pass
    else:
        serialize.pickle = dill
    can_map.pop(FunctionType, None)