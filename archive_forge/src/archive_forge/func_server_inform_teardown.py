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
def server_inform_teardown(self, sreq: 'spb.ServerRequest') -> None:
    request = sreq.inform_teardown
    exit_code = request.exit_code
    self._mux.teardown(exit_code)