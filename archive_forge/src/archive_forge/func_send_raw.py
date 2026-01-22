from __future__ import annotations
import hashlib
import hmac
import json
import logging
import os
import pickle
import pprint
import random
import typing as t
import warnings
from binascii import b2a_hex
from datetime import datetime, timezone
from hmac import compare_digest
import zmq.asyncio
from tornado.ioloop import IOLoop
from traitlets import (
from traitlets.config.configurable import Configurable, LoggingConfigurable
from traitlets.log import get_logger
from traitlets.utils.importstring import import_item
from zmq.eventloop.zmqstream import ZMQStream
from ._version import protocol_version
from .adapter import adapt
from .jsonutil import extract_dates, json_clean, json_default, squash_dates
def send_raw(self, stream: zmq.sugar.socket.Socket, msg_list: list, flags: int=0, copy: bool=True, ident: bytes | list[bytes] | None=None) -> None:
    """Send a raw message via ident path.

        This method is used to send a already serialized message.

        Parameters
        ----------
        stream : ZMQStream or Socket
            The ZMQ stream or socket to use for sending the message.
        msg_list : list
            The serialized list of messages to send. This only includes the
            [p_header,p_parent,p_metadata,p_content,buffer1,buffer2,...] portion of
            the message.
        ident : ident or list
            A single ident or a list of idents to use in sending.
        """
    to_send = []
    if isinstance(ident, bytes):
        ident = [ident]
    if ident is not None:
        to_send.extend(ident)
    to_send.append(DELIM)
    to_send.append(self.sign(msg_list[0:4]))
    to_send.extend(msg_list)
    if isinstance(stream, zmq.asyncio.Socket):
        stream = zmq.Socket.shadow(stream.underlying)
    stream.send_multipart(to_send, flags, copy=copy)