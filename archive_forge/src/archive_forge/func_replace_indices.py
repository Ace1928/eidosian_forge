from pyomo.common.collections import ComponentSet
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.global_set import UnindexedComponent_set
def replace_indices(index, location_set_map, sets):
    """Use `location_set_map` to replace values in `index` with slices
    or an Ellipsis.

    Parameters:
    -----------
    index: `tuple` or scalar
        Index whose values to replace
    location_set_map: `dict`
        Maps locations ("indices") within the index to their
        corresponding set
    sets: `pyomo.common.collections.ComponentSet`
        Contains the sets to replace with slices

    Returns:
    --------
    `tuple`: Index with values replaced by slices

    """
    sets = ComponentSet(sets)
    index = tuple(_to_iterable(index))
    new_index = []
    loc = 0
    len_index = len(index)
    while loc < len_index:
        val = index[loc]
        _set = location_set_map[loc]
        dimen = _set.dimen
        if _set not in sets:
            new_index.append(val)
        elif dimen is not None:
            new_index.append(slice(None, None, None))
        else:
            dimen_none_set = _set
            new_index.append(Ellipsis)
            loc += 1
            while loc < len_index:
                _set = location_set_map[loc]
                if _set is not dimen_none_set:
                    break
                loc += 1
            continue
        loc += 1
    return tuple(new_index)