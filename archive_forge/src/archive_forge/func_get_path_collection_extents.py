import copy
from functools import lru_cache
from weakref import WeakValueDictionary
import numpy as np
import matplotlib as mpl
from . import _api, _path
from .cbook import _to_unmasked_float_array, simple_linear_interpolation
from .bezier import BezierSegment
def get_path_collection_extents(master_transform, paths, transforms, offsets, offset_transform):
    """
    Get bounding box of a `.PathCollection`\\s internal objects.

    That is, given a sequence of `Path`\\s, `.Transform`\\s objects, and offsets, as found
    in a `.PathCollection`, return the bounding box that encapsulates all of them.

    Parameters
    ----------
    master_transform : `~matplotlib.transforms.Transform`
        Global transformation applied to all paths.
    paths : list of `Path`
    transforms : list of `~matplotlib.transforms.Affine2DBase`
        If non-empty, this overrides *master_transform*.
    offsets : (N, 2) array-like
    offset_transform : `~matplotlib.transforms.Affine2DBase`
        Transform applied to the offsets before offsetting the path.

    Notes
    -----
    The way that *paths*, *transforms* and *offsets* are combined follows the same
    method as for collections: each is iterated over independently, so if you have 3
    paths (A, B, C), 2 transforms (α, β) and 1 offset (O), their combinations are as
    follows:

    - (A, α, O)
    - (B, β, O)
    - (C, α, O)
    """
    from .transforms import Bbox
    if len(paths) == 0:
        raise ValueError('No paths provided')
    if len(offsets) == 0:
        _api.warn_deprecated('3.8', message='Calling get_path_collection_extents() with an empty offsets list is deprecated since %(since)s. Support will be removed %(removal)s.')
    extents, minpos = _path.get_path_collection_extents(master_transform, paths, np.atleast_3d(transforms), offsets, offset_transform)
    return Bbox.from_extents(*extents, minpos=minpos)