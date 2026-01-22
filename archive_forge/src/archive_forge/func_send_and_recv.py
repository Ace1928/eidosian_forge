import socket
import struct
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, List, Optional
from wandb.proto import wandb_server_pb2 as spb
from . import tracelog
def send_and_recv(self, *, inform_init: Optional[spb.ServerInformInitRequest]=None, inform_start: Optional[spb.ServerInformStartRequest]=None, inform_attach: Optional[spb.ServerInformAttachRequest]=None, inform_finish: Optional[spb.ServerInformFinishRequest]=None, inform_teardown: Optional[spb.ServerInformTeardownRequest]=None) -> spb.ServerResponse:
    self.send(inform_init=inform_init, inform_start=inform_start, inform_attach=inform_attach, inform_finish=inform_finish, inform_teardown=inform_teardown)
    response = self.read_server_response(timeout=1)
    if response is None:
        raise Exception('No response')
    return response