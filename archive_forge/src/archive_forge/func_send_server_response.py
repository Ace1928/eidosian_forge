import socket
import struct
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, List, Optional
from wandb.proto import wandb_server_pb2 as spb
from . import tracelog
def send_server_response(self, msg: Any) -> None:
    try:
        self._send_message(msg)
    except BrokenPipeError:
        pass