from pyomo.core.base import Block, Reference
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.block import SubclassOf
from pyomo.core.base.set import SetProduct
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from collections import OrderedDict
def flatten_components_along_sets(m, sets, ctype, indices=None, active=None):
    """This function iterates over components (recursively) contained
    in a block and partitions their data objects into components
    indexed only by the specified sets.

    Parameters
    ----------

    m: _BlockData
        Block whose components (and their sub-components) will be
        partitioned

    sets: Tuple of Pyomo Sets
        Sets to be sliced. Returned components will be indexed by
        some combination of these sets, if at all.

    ctype: Subclass of Component
        Type of component to identify and partition

    indices: Iterable or ComponentMap
        Indices of sets to use when descending into subblocks. If an
        iterable is provided, the order corresponds to the order in
        ``sets``. If a ``ComponentMap`` is provided, the keys must be
        in ``sets``.

    active: Bool or None
        If not None, this is a boolean flag used to filter component objects
        by their active status. A reference-to-slice is returned if any data
        object defined by the slice matches this flag.

    Returns
    -------

    List of tuples of Sets, list of lists of Components
        The first entry is a list of tuples of Pyomo Sets. The second is a
        list of lists of Components, indexed by the corresponding sets in
        the first list. If the components are unindexed, ComponentData are
        returned and the tuple of sets contains only UnindexedComponent_set.
        If the components are indexed, they are references-to-slices.

    """
    set_of_sets = ComponentSet(sets)
    if indices is None:
        index_map = ComponentMap()
    elif type(indices) is ComponentMap:
        index_map = indices
    else:
        index_map = ComponentMap(zip(sets, indices))
    for s, idx in index_map.items():
        if idx not in s:
            raise ValueError('%s is a bad index for set %s. \nPlease provide an index that is in the set.' % (idx, s.name))
        if s not in set_of_sets:
            raise RuntimeError('Index specified for set %s that is not one of the sets that will be sliced. Indices should only be provided for sets that will be sliced.' % s.name)
    index_stack = []
    sets_dict = OrderedDict()
    comps_dict = OrderedDict()
    for index_sets, slice_ in generate_sliced_components(m, index_stack, m, set_of_sets, ctype, index_map, active=active):
        key = tuple((id(c) for c in index_sets))
        if key not in sets_dict:
            if len(key) == 0:
                sets_dict[key] = (UnindexedComponent_set,)
            else:
                sets_dict[key] = index_sets
        if key not in comps_dict:
            comps_dict[key] = []
        if len(key) == 0:
            comps_dict[key].append(slice_)
        else:
            slice_.attribute_errors_generate_exceptions = False
            slice_.key_errors_generate_exceptions = False
            comps_dict[key].append(Reference(slice_))
    sets_list = list((sets for sets in sets_dict.values()))
    comps_list = list((comps for comps in comps_dict.values()))
    return (sets_list, comps_list)