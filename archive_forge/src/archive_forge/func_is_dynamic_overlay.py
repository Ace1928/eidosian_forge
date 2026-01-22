import bisect
import re
import traceback
import warnings
from collections import defaultdict, namedtuple
import numpy as np
import param
from packaging.version import Version
from ..core import (
from ..core.ndmapping import item_check
from ..core.operation import Operation
from ..core.options import CallbackError, Cycle
from ..core.spaces import get_nested_streams
from ..core.util import (
from ..element import Points
from ..streams import LinkedStream, Params
from ..util.transform import dim
def is_dynamic_overlay(dmap):
    """
    Traverses a DynamicMap graph and determines if any components
    were overlaid dynamically (i.e. by * on a DynamicMap).
    """
    if not isinstance(dmap, DynamicMap):
        return False
    elif dmap.callback._is_overlay:
        return True
    else:
        return any((is_dynamic_overlay(dm) for dm in dmap.callback.inputs))