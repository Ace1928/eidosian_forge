from __future__ import annotations
import asyncio
import codecs
import collections
import logging
import random
import ssl
import struct
import sys
import time
import uuid
import warnings
from typing import (
from ..datastructures import Headers
from ..exceptions import (
from ..extensions import Extension
from ..frames import (
from ..protocol import State
from ..typing import Data, LoggerLike, Subprotocol
from .compatibility import asyncio_timeout
from .framing import Frame
def broadcast(websockets: Iterable[WebSocketCommonProtocol], message: Data, raise_exceptions: bool=False) -> None:
    """
    Broadcast a message to several WebSocket connections.

    A string (:class:`str`) is sent as a Text_ frame. A bytestring or bytes-like
    object (:class:`bytes`, :class:`bytearray`, or :class:`memoryview`) is sent
    as a Binary_ frame.

    .. _Text: https://www.rfc-editor.org/rfc/rfc6455.html#section-5.6
    .. _Binary: https://www.rfc-editor.org/rfc/rfc6455.html#section-5.6

    :func:`broadcast` pushes the message synchronously to all connections even
    if their write buffers are overflowing. There's no backpressure.

    If you broadcast messages faster than a connection can handle them, messages
    will pile up in its write buffer until the connection times out. Keep
    ``ping_interval`` and ``ping_timeout`` low to prevent excessive memory usage
    from slow connections.

    Unlike :meth:`~websockets.server.WebSocketServerProtocol.send`,
    :func:`broadcast` doesn't support sending fragmented messages. Indeed,
    fragmentation is useful for sending large messages without buffering them in
    memory, while :func:`broadcast` buffers one copy per connection as fast as
    possible.

    :func:`broadcast` skips connections that aren't open in order to avoid
    errors on connections where the closing handshake is in progress.

    :func:`broadcast` ignores failures to write the message on some connections.
    It continues writing to other connections. On Python 3.11 and above, you
    may set ``raise_exceptions`` to :obj:`True` to record failures and raise all
    exceptions in a :pep:`654` :exc:`ExceptionGroup`.

    Args:
        websockets: WebSocket connections to which the message will be sent.
        message: Message to send.
        raise_exceptions: Whether to raise an exception in case of failures.

    Raises:
        TypeError: If ``message`` doesn't have a supported type.

    """
    if not isinstance(message, (str, bytes, bytearray, memoryview)):
        raise TypeError('data must be str or bytes-like')
    if raise_exceptions:
        if sys.version_info[:2] < (3, 11):
            raise ValueError('raise_exceptions requires at least Python 3.11')
        exceptions = []
    opcode, data = prepare_data(message)
    for websocket in websockets:
        if websocket.state is not State.OPEN:
            continue
        if websocket._fragmented_message_waiter is not None:
            if raise_exceptions:
                exception = RuntimeError('sending a fragmented message')
                exceptions.append(exception)
            else:
                websocket.logger.warning('skipped broadcast: sending a fragmented message')
        try:
            websocket.write_frame_sync(True, opcode, data)
        except Exception as write_exception:
            if raise_exceptions:
                exception = RuntimeError('failed to write message')
                exception.__cause__ = write_exception
                exceptions.append(exception)
            else:
                websocket.logger.warning('skipped broadcast: failed to write message', exc_info=True)
    if raise_exceptions:
        raise ExceptionGroup('skipped broadcast', exceptions)