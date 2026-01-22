from pyomo.common.collections import ComponentSet
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.global_set import UnindexedComponent_set
def slice_component_along_sets(comp, sets, context=None):
    """Slice a component along the indices corresponding to some sets,
    wherever they appear in the component's block hierarchy.

    Given a component or component data object, for all parent components
    and parent blocks between the object and the `context` block, replace
    any index corresponding to a set in `sets` with slices or an
    ellipsis.

    Parameters:
    -----------
    comp: `pyomo.core.base.component.Component` or
    `pyomo.core.base.component.ComponentData`
        Component whose parent structure to search and replace
    sets: `pyomo.common.collections.ComponentSet`
        Contains the sets to replace with slices
    context: `pyomo.core.base.block.Block` or
    `pyomo.core.base.block._BlockData`
        Block below which to search for sets

    Returns:
    --------
    `pyomo.core.base.indexed_component_slice.IndexedComponent_slice`:
        Slice of `comp` with wildcards replacing the indices of `sets`

    """
    sets = ComponentSet(sets)
    if context is None:
        context = comp.model()
    call_stack = get_component_call_stack(comp, context)
    comp = context
    sliced_comp = context
    while call_stack:
        call, arg = call_stack.pop()
        if call is IndexedComponent_slice.get_attribute:
            comp = getattr(comp, arg)
            sliced_comp = getattr(sliced_comp, arg)
        elif call is IndexedComponent_slice.get_item:
            index_set = comp.index_set()
            comp = comp[arg]
            location_set_map = get_location_set_map(arg, index_set)
            arg = replace_indices(arg, location_set_map, sets)
            if normalize_index.flatten:
                arg = normalize_index(arg)
            sliced_comp = sliced_comp[arg]
    return sliced_comp