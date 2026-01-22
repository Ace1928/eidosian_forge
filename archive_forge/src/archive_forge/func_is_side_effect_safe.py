import collections
from enum import Enum
from typing import Any, Callable, Dict, List
from .. import variables
from ..current_scope_id import current_scope_id
from ..exc import unimplemented
from ..source import AttrSource, Source
from ..utils import identity, istype
def is_side_effect_safe(m: MutableLocalBase):
    scope_id = current_scope_id()
    if _is_top_level_scope(scope_id):
        return True
    return m.scope == scope_id