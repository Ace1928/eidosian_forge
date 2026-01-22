from pyomo.core.base.componentuid import ComponentUID
from pyomo.util.slices import slice_component_along_sets
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.dae.flatten import get_slice_for_set
Attempt to convert the provided "var" object into a CUID with wildcards

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

    