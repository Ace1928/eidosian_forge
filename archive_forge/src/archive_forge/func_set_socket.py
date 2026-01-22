import socket
import struct
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, List, Optional
from wandb.proto import wandb_server_pb2 as spb
from . import tracelog
def set_socket(self, sock: socket.socket) -> None:
    self._sock = sock
    self._detect_bufsize()