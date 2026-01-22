import sys
import pprint as _pprint_
from pyomo.common.collections import ComponentMap
import pyomo.core
from pyomo.core.expr.numvalue import NumericValue
from pyomo.core.kernel.base import (
def generate_names(node, convert=str, prefix='', **kwds):
    """
    Generate names relative to this object for all
    objects stored under it.

    This function is useful in situations where names
    are used often, but they do not need to be
    dynamically regenerated each time.

    Args:
        node: The root object below which names are
            generated.
        convert (function): A function that converts a
            storage key into a string
            representation. Default is str.
        prefix (str): A string to prefix names with.
        **kwds: Additional keywords passed to the
            preorder_traversal function.

    Returns:
        A component map that behaves as a dictionary
        mapping objects to names.
    """
    traversal = preorder_traversal(node, **kwds)
    names = ComponentMap()
    try:
        next(traversal)
    except StopIteration:
        return names
    for obj in traversal:
        parent = obj.parent
        name = parent._child_storage_entry_string % convert(obj.storage_key)
        if parent is not node:
            names[obj] = names[parent] + parent._child_storage_delimiter_string + name
        else:
            names[obj] = prefix + name
    return names