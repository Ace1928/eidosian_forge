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
def compute_sizes(sizes, size_fn, scaling_factor, scaling_method, base_size):
    """
    Scales point sizes according to a scaling factor,
    base size and size_fn, which will be applied before
    scaling.
    """
    if sizes.dtype.kind not in ('i', 'f'):
        return None
    if scaling_method == 'area':
        pass
    elif scaling_method == 'width':
        scaling_factor = scaling_factor ** 2
    else:
        raise ValueError(f'Invalid value for argument "scaling_method": "{scaling_method}". Valid values are: "width", "area".')
    sizes = size_fn(sizes)
    return base_size * scaling_factor * sizes