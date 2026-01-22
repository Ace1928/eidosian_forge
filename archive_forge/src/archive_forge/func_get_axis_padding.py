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
def get_axis_padding(padding):
    """
    Process a padding value supplied as a tuple or number and returns
    padding values for x-, y- and z-axis.
    """
    if isinstance(padding, tuple):
        if len(padding) == 2:
            xpad, ypad = padding
            zpad = 0
        elif len(padding) == 3:
            xpad, ypad, zpad = padding
        else:
            raise ValueError('Padding must be supplied as an number applied to all axes or a length two or three tuple corresponding to the x-, y- and optionally z-axis')
    else:
        xpad, ypad, zpad = (padding,) * 3
    return (xpad, ypad, zpad)