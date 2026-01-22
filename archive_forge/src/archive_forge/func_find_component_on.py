import codecs
import re
import ply.lex
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import pickle
from pyomo.common.deprecation import deprecated
from pyomo.core.base.component_namer import (
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.reference import Reference
def find_component_on(self, block):
    """
        Return the (unique) component in the block.  If the CUID contains
        a wildcard in the last component, then returns that component.  If
        there are wildcards elsewhere (or the last component was a partial
        slice), then returns a reference.  See also list_components below.
        """
    obj = self._resolve_cuid(block)
    if isinstance(obj, IndexedComponent_slice):
        obj.key_errors_generate_exceptions = False
        obj.attribute_errors_generate_exceptions = False
        obj = Reference(obj)
        try:
            next(iter(obj))
        except StopIteration:
            obj = None
    return obj