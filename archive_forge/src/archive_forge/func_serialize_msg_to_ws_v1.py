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
def serialize_msg_to_ws_v1(msg_or_list, channel, pack=None):
    """Serialize a message using the v1 protocol."""
    if pack:
        msg_list = [pack(msg_or_list['header']), pack(msg_or_list['parent_header']), pack(msg_or_list['metadata']), pack(msg_or_list['content'])]
    else:
        msg_list = msg_or_list
    channel = channel.encode('utf-8')
    offsets: List[Any] = []
    offsets.append(8 * (1 + 1 + len(msg_list) + 1))
    offsets.append(len(channel) + offsets[-1])
    for msg in msg_list:
        offsets.append(len(msg) + offsets[-1])
    offset_number = len(offsets).to_bytes(8, byteorder='little')
    offsets = [offset.to_bytes(8, byteorder='little') for offset in offsets]
    bin_msg = b''.join([offset_number, *offsets, channel, *msg_list])
    return bin_msg