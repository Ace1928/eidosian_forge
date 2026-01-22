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
def was_already_deleted(self, state):
    """Return ``True`` if the given state is expired and was deleted
        previously.
        """
    if state.expired:
        try:
            state._load_expired(state, attributes.PASSIVE_OFF)
        except orm_exc.ObjectDeletedError:
            self.session._remove_newly_deleted([state])
            return True
    return False