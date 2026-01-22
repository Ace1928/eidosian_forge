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
def filter_states_for_dep(self, dep, states):
    """Filter the given list of InstanceStates to those relevant to the
        given DependencyProcessor.

        """
    mapper_for_dep = self._mapper_for_dep
    return [s for s in states if mapper_for_dep[s.manager.mapper, dep]]