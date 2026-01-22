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
def server_inform_attach(self, sreq: 'spb.ServerRequest') -> None:
    request = sreq.inform_attach
    stream_id = request._info.stream_id
    self._clients.add_client(self._sock_client)
    inform_attach_response = spb.ServerInformAttachResponse()
    inform_attach_response.settings.CopyFrom(self._mux._streams[stream_id]._settings._proto)
    response = spb.ServerResponse(inform_attach_response=inform_attach_response)
    self._sock_client.send_server_response(response)
    iface = self._mux.get_stream(stream_id).interface
    assert iface