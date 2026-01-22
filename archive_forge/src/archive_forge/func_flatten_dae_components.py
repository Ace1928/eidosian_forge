from pyomo.core.base import Block, Reference
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.block import SubclassOf
from pyomo.core.base.set import SetProduct
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from collections import OrderedDict
def flatten_dae_components(model, time, ctype, indices=None, active=None):
    """Partitions components into ComponentData and Components indexed only
    by the provided set.

    Parameters
    ----------

    model: _BlockData
        Block whose components are partitioned

    time: Set
        Indexing by this set (and only this set) will be preserved in the
        returned components.

    ctype: Subclass of Component
        Type of component to identify, partition, and return

    indices: Tuple or ComponentMap
        Contains the index of the specified set to be used when descending
        into blocks

    active: Bool or None
        If provided, used as a filter to only return components with the
        specified active flag. A reference-to-slice is returned if any
        data object defined by the slice matches this flag.

    Returns
    -------
    List of ComponentData, list of Component
        The first list contains ComponentData for all components not
        indexed by the provided set. The second contains references-to
        -slices for all components indexed by the provided set.

    """
    target = ComponentSet((time,))
    sets_list, comps_list = flatten_components_along_sets(model, target, ctype, indices=indices, active=active)
    scalar_comps = []
    dae_comps = []
    for sets, comps in zip(sets_list, comps_list):
        if len(sets) == 1 and sets[0] is time:
            dae_comps = comps
        elif len(sets) == 0 or (len(sets) == 1 and sets[0] is UnindexedComponent_set):
            scalar_comps = comps
        else:
            raise RuntimeError('Invalid model for `flatten_dae_components`.\nThis can happen if your model has components that are\nindexed by time (explicitly or implicitly) multiple times.')
    return (scalar_comps, dae_comps)