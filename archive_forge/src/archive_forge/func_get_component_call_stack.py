from pyomo.common.collections import ComponentSet
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.global_set import UnindexedComponent_set
def get_component_call_stack(comp, context=None):
    """Get the call stack necessary to locate a `Component`

    The call stack is a `list` of `tuple`s where the first entry is a
    code for `__getattr__` or  `__getitem__`, using the same convention
    as `IndexedComponent_slice`. The second entry is the argument of
    the corresponding function. Following this sequence of calls from
    `context` (or the top-level model if `context is None`) will
    produce comp.

    Parameters:
    -----------
    comp : `pyomo.core.base.component.Component`
        The component to locate
    context : `pyomo.core.base.block.Block`
        The block within which to locate the component. If `None`, the
        top-level model will be used.

    Returns:
    --------
    `list` : Contains the necessary method calls and their arguments.
             Note that the calls should be applied in reverse order.
             This is the opposite direction as in IndexedComponent_slice.

    """
    call_stack = []
    while comp.parent_block() is not None:
        if comp is context:
            break
        parent_component = comp.parent_component()
        if parent_component.is_indexed() and parent_component is not comp:
            call_stack.append((IndexedComponent_slice.get_item, comp.index()))
        if parent_component is context:
            break
        call_stack.append((IndexedComponent_slice.get_attribute, parent_component.local_name))
        comp = comp.parent_block()
    return call_stack