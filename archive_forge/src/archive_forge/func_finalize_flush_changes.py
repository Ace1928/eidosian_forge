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
def finalize_flush_changes(self) -> None:
    """Mark processed objects as clean / deleted after a successful
        flush().

        This method is called within the flush() method after the
        execute() method has succeeded and the transaction has been committed.

        """
    if not self.states:
        return
    states = set(self.states)
    isdel = {s for s, (isdelete, listonly) in self.states.items() if isdelete}
    other = states.difference(isdel)
    if isdel:
        self.session._remove_newly_deleted(isdel)
    if other:
        self.session._register_persistent(other)