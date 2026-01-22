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
def get_dynamic_mode(composite):
    """Returns the common mode of the dynamic maps in given composite object"""
    dynmaps = composite.traverse(lambda x: x, [DynamicMap])
    holomaps = composite.traverse(lambda x: x, ['HoloMap'])
    dynamic_unbounded = any((m.unbounded for m in dynmaps))
    if holomaps:
        validate_unbounded_mode(holomaps, dynmaps)
    elif dynamic_unbounded and (not holomaps):
        raise Exception('DynamicMaps in unbounded mode must be displayed alongside a HoloMap to define the sampling.')
    return (dynmaps and (not holomaps), dynamic_unbounded)