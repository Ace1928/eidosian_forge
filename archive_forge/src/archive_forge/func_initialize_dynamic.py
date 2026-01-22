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
def initialize_dynamic(obj):
    """
    Initializes all DynamicMap objects contained by the object
    """
    dmaps = obj.traverse(lambda x: x, specs=[DynamicMap])
    for dmap in dmaps:
        if dmap.unbounded:
            continue
        if not len(dmap):
            dmap[dmap._initial_key()]