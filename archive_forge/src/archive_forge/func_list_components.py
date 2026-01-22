import codecs
import re
import ply.lex
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import pickle
from pyomo.common.deprecation import deprecated
from pyomo.core.base.component_namer import (
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.reference import Reference
def list_components(self, block):
    """Generator returning all components matching this ComponentUID"""
    obj = self._resolve_cuid(block)
    if obj is None:
        return
    if isinstance(obj, IndexedComponent_slice):
        obj.key_errors_generate_exceptions = False
        obj.attribute_errors_generate_exceptions = False
        for o in obj:
            yield o
    else:
        yield obj