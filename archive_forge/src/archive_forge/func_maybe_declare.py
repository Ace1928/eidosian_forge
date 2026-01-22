from __future__ import annotations
from itertools import count
from typing import TYPE_CHECKING
from .common import maybe_declare
from .compression import compress
from .connection import is_connection, maybe_channel
from .entity import Exchange, Queue, maybe_delivery_mode
from .exceptions import ContentDisallowed
from .serialization import dumps, prepare_accept_content
from .utils.functional import ChannelPromise, maybe_list
def maybe_declare(self, entity, retry=False, **retry_policy):
    """Declare exchange if not already declared during this session."""
    if entity:
        return maybe_declare(entity, self.channel, retry, **retry_policy)