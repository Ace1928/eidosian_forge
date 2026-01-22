import queue
import socket
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional
from wandb.proto import wandb_server_pb2 as spb
from wandb.sdk.internal.settings_static import SettingsStatic
from ..lib import tracelog
from ..lib.sock_client import SockClient, SockClientClosedError
from .streams import StreamMux
def server_record_publish(self, sreq: 'spb.ServerRequest') -> None:
    record = sreq.record_publish
    record.control.relay_id = self._sock_client._sockid
    stream_id = record._info.stream_id
    iface = self._mux.get_stream(stream_id).interface
    assert iface.record_q
    iface.record_q.put(record)