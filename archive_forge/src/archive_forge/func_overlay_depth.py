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
def overlay_depth(obj):
    """
    Computes the depth of a DynamicMap overlay if it can be determined
    otherwise return None.
    """
    if isinstance(obj, DynamicMap):
        if isinstance(obj.last, CompositeOverlay):
            return len(obj.last)
        elif obj.last is None:
            return None
        return 1
    else:
        return 1