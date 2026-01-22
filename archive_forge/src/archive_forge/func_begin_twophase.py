from __future__ import annotations
import typing
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union
from .base import Connection
from .base import Engine
from .interfaces import ConnectionEventsTarget
from .interfaces import DBAPIConnection
from .interfaces import DBAPICursor
from .interfaces import Dialect
from .. import event
from .. import exc
from ..util.typing import Literal
def begin_twophase(self, conn: Connection, xid: Any) -> None:
    """Intercept begin_twophase() events.

        :param conn: :class:`_engine.Connection` object
        :param xid: two-phase XID identifier

        """