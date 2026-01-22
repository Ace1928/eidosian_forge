from __future__ import annotations
from typing import Any
from typing import Dict
from typing import Optional
from typing import Set
from typing import TYPE_CHECKING
from . import attributes
from . import exc as orm_exc
from . import util as orm_util
from .. import event
from .. import util
from ..util import topological
def states_for_mapper_hierarchy(self, mapper, isdelete, listonly):
    checktup = (isdelete, listonly)
    for mapper in mapper.base_mapper.self_and_descendants:
        for state in self.mappers[mapper]:
            if self.states[state] == checktup:
                yield state