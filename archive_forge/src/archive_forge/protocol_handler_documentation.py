from __future__ import annotations
import logging # isort:skip
from typing import Any, Callable
from ..protocol.exceptions import ProtocolError
from .session import ServerSession
 Delegate a received message to the appropriate handler.

        Args:
            message (Message) :
                The message that was receive that needs to be handled

            connection (ServerConnection) :
                The connection that received this message

        Raises:
            ProtocolError

        