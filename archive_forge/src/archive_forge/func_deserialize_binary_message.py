import json
import struct
from typing import Any, List
from jupyter_client.session import Session
from tornado.websocket import WebSocketHandler
from traitlets import Float, Instance, Unicode, default
from traitlets.config import LoggingConfigurable
from jupyter_client.jsonutil import extract_dates
from jupyter_server.transutils import _i18n
from .abc import KernelWebsocketConnectionABC
def deserialize_binary_message(bmsg):
    """deserialize a message from a binary blog

    Header:

    4 bytes: number of msg parts (nbufs) as 32b int
    4 * nbufs bytes: offset for each buffer as integer as 32b int

    Offsets are from the start of the buffer, including the header.

    Returns
    -------
    message dictionary
    """
    nbufs = struct.unpack('!i', bmsg[:4])[0]
    offsets = list(struct.unpack('!' + 'I' * nbufs, bmsg[4:4 * (nbufs + 1)]))
    offsets.append(None)
    bufs = []
    for start, stop in zip(offsets[:-1], offsets[1:]):
        bufs.append(bmsg[start:stop])
    msg = json.loads(bufs[0].decode('utf8'))
    msg['header'] = extract_dates(msg['header'])
    msg['parent_header'] = extract_dates(msg['parent_header'])
    msg['buffers'] = bufs[1:]
    return msg