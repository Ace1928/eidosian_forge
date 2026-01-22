import socket
import struct
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, List, Optional
from wandb.proto import wandb_server_pb2 as spb
from . import tracelog
Read full message from socket.

        Args:
            timeout: number of seconds to wait on socket data.

        Raises:
            SockClientClosedError: socket has been closed.
        