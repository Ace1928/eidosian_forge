from pyomo.core.base.componentuid import ComponentUID
from pyomo.util.slices import slice_component_along_sets
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.dae.flatten import get_slice_for_set
def get_indexed_cuid(var, sets=None, dereference=None, context=None):
    """Attempt to convert the provided "var" object into a CUID with wildcards

    Arguments
    ---------
    var:
        Object to process. May be a VarData, IndexedVar (reference or otherwise),
        ComponentUID, slice, or string.
    sets: Tuple of sets
        Sets to use if slicing a vardata object
    dereference: None or int
        Number of times we may access referent attribute to recover a
        "base component" from a reference.
    context: Block
        Block with respect to which slices and CUIDs will be generated

    Returns
    -------
    ``ComponentUID``
        ComponentUID corresponding to the provided ``var`` and sets

    """
    if isinstance(var, ComponentUID):
        return var
    elif isinstance(var, (str, IndexedComponent_slice)):
        return ComponentUID(var, context=context)
    if dereference is None:
        remaining_dereferences = int(var.parent_block() is None)
    else:
        remaining_dereferences = int(dereference)
    if var.is_indexed():
        if var.is_reference() and remaining_dereferences:
            remaining_dereferences -= 1
            referent = var.referent
            if isinstance(referent, IndexedComponent_slice):
                return ComponentUID(referent, context=context)
            else:
                dereference = dereference if dereference is None else remaining_dereferences
                return get_indexed_cuid(referent, sets, dereference=dereference)
        else:
            index = tuple((get_slice_for_set(s) for s in var.index_set().subsets()))
            return ComponentUID(var[index], context=context)
    else:
        if sets is None:
            raise ValueError('A ComponentData %s was provided but no set. We need to know\nwhat set this component should be indexed by.' % var.name)
        slice_ = slice_component_along_sets(var, sets)
        return ComponentUID(slice_, context=context)