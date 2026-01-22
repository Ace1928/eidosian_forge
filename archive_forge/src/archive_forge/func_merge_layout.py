import copy
import re
import numpy as np
from plotly import colors
from ...core.util import isfinite, max_range
from ..util import color_intervals, process_cmap
def merge_layout(obj, subobj):
    """
    Merge layout objects recursively

    Note: This function mutates the input obj dict, but it does not mutate
    the subobj dict

    Parameters
    ----------
    obj: dict
        dict into which the sub-figure dict will be merged
    subobj: dict
        dict that sill be copied and merged into `obj`
    """
    for prop, val in subobj.items():
        if isinstance(val, dict) and prop in obj:
            merge_layout(obj[prop], val)
        elif isinstance(val, list) and obj.get(prop, None) and isinstance(obj[prop][0], dict):
            obj[prop].extend(val)
        elif prop == 'style' and val == 'white-bg' and obj.get('style', None):
            pass
        elif val is not None:
            obj[prop] = copy.deepcopy(val)