from __future__ import annotations
import struct
from typing import Awaitable, overload
import zmq
import zmq.asyncio
from zmq._typing import TypedDict
from zmq.error import _check_version
def parse_monitor_message(msg: list[bytes]) -> _MonitorMessage:
    """decode zmq_monitor event messages.

    Parameters
    ----------
    msg : list(bytes)
        zmq multipart message that has arrived on a monitor PAIR socket.

        First frame is::

            16 bit event id
            32 bit event value
            no padding

        Second frame is the endpoint as a bytestring

    Returns
    -------
    event : dict
        event description as dict with the keys `event`, `value`, and `endpoint`.
    """
    if len(msg) != 2 or len(msg[0]) != 6:
        raise RuntimeError('Invalid event message format: %s' % msg)
    event_id, value = struct.unpack('=hi', msg[0])
    event: _MonitorMessage = {'event': zmq.Event(event_id), 'value': zmq.Event(value), 'endpoint': msg[1]}
    return event