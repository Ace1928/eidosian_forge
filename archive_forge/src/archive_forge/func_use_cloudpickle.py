import copy
import pickle
import sys
import typing
import warnings
from types import FunctionType
from traitlets.log import get_logger
from traitlets.utils.importstring import import_item
def use_cloudpickle():
    """use cloudpickle to expand serialization support

    adds support for object methods and closures to serialization.
    """
    import cloudpickle
    global pickle
    pickle = cloudpickle
    try:
        from ipykernel import serialize
    except ImportError:
        pass
    else:
        serialize.pickle = cloudpickle
    can_map.pop(FunctionType, None)