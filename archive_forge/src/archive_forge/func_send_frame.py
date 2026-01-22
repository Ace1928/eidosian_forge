from __future__ import annotations
import enum
import logging
import uuid
from typing import Generator, List, Optional, Type, Union
from .exceptions import (
from .extensions import Extension
from .frames import (
from .http11 import Request, Response
from .streams import StreamReader
from .typing import LoggerLike, Origin, Subprotocol
def send_frame(self, frame: Frame) -> None:
    if self.state is not OPEN:
        raise InvalidState(f'cannot write to a WebSocket in the {self.state.name} state')
    if self.debug:
        self.logger.debug('> %s', frame)
    self.writes.append(frame.serialize(mask=self.side is CLIENT, extensions=self.extensions))