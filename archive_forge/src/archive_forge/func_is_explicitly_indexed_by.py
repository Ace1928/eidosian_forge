from collections import Counter
from pyomo.common.collections import ComponentSet
from pyomo.core.base import Constraint, Block
from pyomo.core.base.set import SetProduct
def is_explicitly_indexed_by(comp, *sets, **kwargs):
    """
    Function for determining whether a pyomo component is indexed by a
    set or group of sets.

    Args:
        comp : Some Pyomo component, possibly indexed
        sets : Pyomo Sets to check indexing by
        expand_all_set_operators : Whether or not to expand all set operators
                                   in the subsets method

    Returns:
        A bool that is True if comp is directly indexed by every set in sets.
    """
    if not comp.is_indexed():
        return False
    for s in sets:
        if isinstance(s, SetProduct):
            msg = 'Checking for explicit indexing by a SetProduct is not supported'
            raise TypeError(msg)
    expand_all_set_operators = kwargs.pop('expand_all_set_operators', False)
    if kwargs:
        keys = kwargs.keys()
        raise ValueError('Unrecognized keyword arguments: %s' % str(keys))
    projected_subsets = comp.index_set().subsets(expand_all_set_operators=expand_all_set_operators)
    subset_set = ComponentSet(projected_subsets)
    return all([_ in subset_set for _ in sets])