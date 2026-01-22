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
def compute_overlayable_zorders(obj, path=None):
    """
    Traverses an overlayable composite container to determine which
    objects are associated with specific (Nd)Overlay layers by
    z-order, making sure to take DynamicMap Callables into
    account. Returns a mapping between the zorders of each layer and a
    corresponding lists of objects.

    Used to determine which overlaid subplots should be linked with
    Stream callbacks.
    """
    if path is None:
        path = []
    path = path + [obj]
    zorder_map = defaultdict(list)
    if not isinstance(obj, DynamicMap):
        if isinstance(obj, CompositeOverlay):
            for z, o in enumerate(obj):
                zorder_map[z] = [o, obj]
        elif isinstance(obj, HoloMap):
            for el in obj.values():
                if isinstance(el, CompositeOverlay):
                    for k, v in compute_overlayable_zorders(el, path).items():
                        zorder_map[k] += v + [obj]
                else:
                    zorder_map[0] += [obj, el]
        elif obj not in zorder_map[0]:
            zorder_map[0].append(obj)
        return zorder_map
    isoverlay = isinstance(obj.last, CompositeOverlay)
    isdynoverlay = obj.callback._is_overlay
    if obj not in zorder_map[0] and (not isoverlay):
        zorder_map[0].append(obj)
    depth = overlay_depth(obj)
    dmap_inputs = obj.callback.inputs if obj.callback.link_inputs else []
    for z, inp in enumerate(dmap_inputs):
        no_zorder_increment = False
        if any((not (isoverlay_fn(p) or p.last is None) for p in path)) and isoverlay_fn(inp):
            no_zorder_increment = True
        input_depth = overlay_depth(inp)
        if depth is not None and input_depth is not None and (depth < input_depth):
            if depth > 1:
                continue
            else:
                no_zorder_increment = True
        z = z if isdynoverlay else 0
        deep_zorders = compute_overlayable_zorders(inp, path=path)
        offset = max(zorder_map.keys())
        for dz, objs in deep_zorders.items():
            global_z = offset + z if no_zorder_increment else offset + dz + z
            zorder_map[global_z] = list(unique_iterator(zorder_map[global_z] + objs))
    found = any((isinstance(p, DynamicMap) and p.callback._is_overlay for p in path))
    linked = any((isinstance(s, (LinkedStream, Params)) and s.linked for s in obj.streams))
    if (found or linked) and isoverlay and (not isdynoverlay):
        offset = max(zorder_map.keys())
        for z, o in enumerate(obj.last):
            if isoverlay and linked:
                zorder_map[offset + z].append(obj)
            if o not in zorder_map[offset + z]:
                zorder_map[offset + z].append(o)
    return zorder_map