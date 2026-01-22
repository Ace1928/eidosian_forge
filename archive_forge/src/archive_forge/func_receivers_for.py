from __future__ import annotations
import typing as t
from collections import defaultdict
from contextlib import contextmanager
from inspect import iscoroutinefunction
from warnings import warn
from weakref import WeakValueDictionary
from blinker._utilities import annotatable_weakref
from blinker._utilities import hashable_identity
from blinker._utilities import IdentityType
from blinker._utilities import lazy_property
from blinker._utilities import reference
from blinker._utilities import symbol
from blinker._utilities import WeakTypes
def receivers_for(self, sender: t.Any) -> t.Generator[t.Callable[[t.Any], t.Any], None, None]:
    """Iterate all live receivers listening for *sender*."""
    if self.receivers:
        sender_id = hashable_identity(sender)
        if sender_id in self._by_sender:
            ids = self._by_sender[ANY_ID] | self._by_sender[sender_id]
        else:
            ids = self._by_sender[ANY_ID].copy()
        for receiver_id in ids:
            receiver = self.receivers.get(receiver_id)
            if receiver is None:
                continue
            if isinstance(receiver, WeakTypes):
                strong = receiver()
                if strong is None:
                    self._disconnect(receiver_id, ANY_ID)
                    continue
                receiver = strong
            yield receiver