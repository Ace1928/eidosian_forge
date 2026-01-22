import codecs
import re
import ply.lex
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import pickle
from pyomo.common.deprecation import deprecated
from pyomo.core.base.component_namer import (
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.reference import Reference
@ply.lex.TOKEN('[a-zA-Z_0-9][^' + re.escape(special_chars) + ']*')
def t_WORD(t):
    t.value = t.value.strip()
    return t