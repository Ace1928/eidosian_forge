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
def register_cmaps(category, provider, source, bg, names):
    """
    Maintain descriptions of colormaps that include the following information:

    name     - string name for the colormap
    category - intended use or purpose, mostly following matplotlib
    provider - package providing the colormap directly
    source   - original source or creator of the colormaps
    bg       - base/background color expected for the map
               ('light','dark','medium','any' (unknown or N/A))
    """
    for name in names:
        bisect.insort(cmap_info, CMapInfo(name=name, provider=provider, category=category, source=source, bg=bg))