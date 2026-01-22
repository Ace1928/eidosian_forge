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
def dim_range_key(eldim):
    """
    Returns the key to look up a dimension range.
    """
    if isinstance(eldim, dim):
        dim_name = repr(eldim)
        if dim_name.startswith("dim('") and dim_name.endswith("')"):
            dim_name = dim_name[5:-2]
    else:
        dim_name = eldim.name
    return dim_name