from collections import Counter
from pyomo.common.collections import ComponentSet
from pyomo.core.base import Constraint, Block
from pyomo.core.base.set import SetProduct
def is_in_block_indexed_by(comp, s, stop_at=None):
    """
    Function for determining whether a component is contained in a
    block that is indexed by a particular set.

    Args:
        comp : Component whose parent blocks are checked
        s : Set for which indices are checked
        stop_at : Block at which to stop searching if reached, regardless
                  of whether or not it is indexed by s

    Returns:
        Bool that is true if comp is contained in a block indexed by s
    """
    parent = comp.parent_block()
    while parent is not None:
        if parent is stop_at:
            return False
        parent = parent.parent_component()
        if parent is stop_at:
            return False
        if is_explicitly_indexed_by(parent, s):
            return True
        else:
            parent = parent.parent_block()
    return False