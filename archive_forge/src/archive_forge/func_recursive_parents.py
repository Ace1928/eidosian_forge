import collections
from enum import Enum
from typing import Any, Callable, Dict, List
from .. import variables
from ..current_scope_id import current_scope_id
from ..exc import unimplemented
from ..source import AttrSource, Source
from ..utils import identity, istype
def recursive_parents(self):
    rv = dict(self.parents)
    worklist = list(self.parents)
    while worklist:
        for parent in worklist.pop().parents:
            if parent not in rv:
                assert isinstance(parent, ParentsTracker)
                rv[parent] = True
                worklist.append(parent)
    return rv.keys()