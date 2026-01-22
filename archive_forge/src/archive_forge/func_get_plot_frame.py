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
def get_plot_frame(map_obj, key_map, cached=False):
    """Returns the current frame in a mapping given a key mapping.

    Args:
        obj: Nested Dimensioned object
        key_map: Dictionary mapping between dimensions and key value
        cached: Whether to allow looking up key in cache

    Returns:
        The item in the mapping corresponding to the supplied key.
    """
    if map_obj.kdims and len(map_obj.kdims) == 1 and (map_obj.kdims[0] == 'Frame') and (not isinstance(map_obj, DynamicMap)):
        return map_obj.last
    key = tuple((key_map[kd.name] for kd in map_obj.kdims if kd.name in key_map))
    if key in map_obj.data and cached:
        return map_obj.data[key]
    else:
        try:
            return map_obj[key]
        except KeyError:
            return None
        except (StopIteration, CallbackError) as e:
            raise e
        except Exception:
            print(traceback.format_exc())
            return None