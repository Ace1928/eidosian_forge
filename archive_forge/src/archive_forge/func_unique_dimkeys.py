from collections import defaultdict
from operator import itemgetter
from .dimension import Dimension
from .util import merge_dimensions
def unique_dimkeys(obj, default_dim='Frame'):
    """
    Finds all common dimension keys in the object including subsets of
    dimensions. If there are is no common subset of dimensions, None
    is returned.

    Returns the list of dimensions followed by the list of unique
    keys.
    """
    from .ndmapping import NdMapping, item_check
    from .spaces import HoloMap
    key_dims = obj.traverse(lambda x: (tuple(x.kdims), list(x.data.keys())), (HoloMap,))
    if not key_dims:
        return ([Dimension(default_dim)], [(0,)])
    dim_groups, keys = zip(*sorted(key_dims, key=lambda x: -len(x[0])))
    dgroups = [frozenset((d.name for d in dg)) for dg in dim_groups]
    subset = all((g1 <= g2 or g1 >= g2 for g1 in dgroups for g2 in dgroups))
    if subset:
        dims = merge_dimensions(dim_groups)
        all_dims = sorted(dims, key=lambda x: dim_groups[0].index(x))
    else:
        hmaps = obj.traverse(lambda x: x, ['HoloMap'])
        if hmaps:
            raise ValueError('When combining HoloMaps into a composite plot their dimensions must be subsets of each other.')
        dimensions = merge_dimensions(dim_groups)
        dim_keys = {}
        for dims, keys in key_dims:
            for key in keys:
                for d, k in zip(dims, key):
                    dim_keys[d.name] = k
        if dim_keys:
            keys = [tuple((dim_keys.get(dim.name) for dim in dimensions))]
        else:
            keys = []
        return (merge_dimensions(dim_groups), keys)
    ndims = len(all_dims)
    unique_keys = []
    for group, subkeys in zip(dim_groups, keys):
        dim_idxs = [all_dims.index(dim) for dim in group]
        for key in subkeys:
            padded_key = create_ndkey(ndims, dim_idxs, key)
            matches = [item for item in unique_keys if padded_key == tuple((k if k is None else i for i, k in zip(item, padded_key)))]
            if not matches:
                unique_keys.append(padded_key)
    with item_check(False):
        sorted_keys = NdMapping({key: None for key in unique_keys}, kdims=all_dims).data.keys()
    return (all_dims, list(sorted_keys))