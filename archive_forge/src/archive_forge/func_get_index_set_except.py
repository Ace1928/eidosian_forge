from collections import Counter
from pyomo.common.collections import ComponentSet
from pyomo.core.base import Constraint, Block
from pyomo.core.base.set import SetProduct
def get_index_set_except(comp, *sets):
    """
    Function for getting indices of a component over a product of its
    indexing sets other than those specified. Indices for the specified
    sets can be used to construct indices of the proper dimension for the
    original component via the index_getter function.

    Args:
        comp : Component whose indexing sets are to be manipulated
        sets : Sets to omit from the set_except product

    Returns:
        A dictionary. Maps 'set_except' to a Pyomo Set or SetProduct
        of comp's index set, excluding those in sets. Maps
        'index_getter' to a function that returns an index of the
        proper dimension for comp, given an element of set_except
        and a value for each set excluded. These values must be provided
        in the same order their Sets were provided in the sets argument.
    """
    if not is_explicitly_indexed_by(comp, *sets):
        msg = comp.name + ' is not indexed by at least one of ' + str([s.name for s in sets])
        raise ValueError(msg)
    return get_indices_of_projection(comp.index_set(), *sets)