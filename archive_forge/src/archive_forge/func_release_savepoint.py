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
def release_savepoint(self, conn: Connection, name: str, context: None) -> None:
    """Intercept release_savepoint() events.

        :param conn: :class:`_engine.Connection` object
        :param name: specified name used for the savepoint.
        :param context: not used

        """