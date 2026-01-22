import socket
import struct
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, List, Optional
from wandb.proto import wandb_server_pb2 as spb
from . import tracelog
def read_server_response(self, timeout: Optional[int]=None) -> Optional[spb.ServerResponse]:
    data = self._read_packet_bytes(timeout=timeout)
    if not data:
        return None
    rec = spb.ServerResponse()
    rec.ParseFromString(data)
    tracelog.log_message_recv(rec, self._sockid)
    return rec