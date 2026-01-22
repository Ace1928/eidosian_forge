import collections
from enum import Enum
from typing import Any, Callable, Dict, List
from .. import variables
from ..current_scope_id import current_scope_id
from ..exc import unimplemented
from ..source import AttrSource, Source
from ..utils import identity, istype
def recursive_realize(self):
    """Realize all objects under this"""
    return VariableTracker.apply(lambda x: x.realize(), self)